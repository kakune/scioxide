//! Locally optimal block preconditioned eigeniteration with B-orthogonal constraints.
use super::workspace::{LobpcgLayout, LobpcgRequirements};
use super::{ActionSlot, SpectralTermination, Work, run_action};
use crate::kernel::subspace::{self, Columns};
use crate::kernel::{finite, level2};
use crate::workspace::SolveSlot;
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};

// Match the long Krylov recurrence verification interval.
const VERIFICATION_INTERVAL: usize = 50;
use crate::{
    HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    LinalgWorkspace, Matrix, VectorView,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    AttemptRecord, BlockVectorView, Budget, LinearOperator, NumericScalar, Preconditioner,
    RealScalar, SciOxideError, Tolerance,
};
/// End of the spectrum whose eigenvalues [`LobpcgSolver`] computes.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{LobpcgSolver, LobpcgSelection};
/// let options = LobpcgSolver::<f64>::new(2).with_selection(LobpcgSelection::Largest);
/// assert_eq!(options.selection(), LobpcgSelection::Largest);
/// ```
///
/// The magnitude orders of [`SpectrumSelection`](crate::SpectrumSelection) are
/// not accepted:
///
/// ```compile_fail
/// use scioxide_linalg::{LobpcgSolver, SpectrumSelection};
/// LobpcgSolver::<f64>::new(1).with_selection(SpectrumSelection::LargestMagnitude);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LobpcgSelection {
    /// Smallest algebraic eigenvalues.
    #[default]
    Smallest,
    /// Largest algebraic eigenvalues.
    Largest,
}

/// Solver for extreme eigenpairs of a Hermitian pencil `A x = λ B x` by LOBPCG.
///
/// `A` must be Hermitian and `B` (the identity by default) Hermitian positive
/// definite. Both are used only through products, so they can be matrix-free;
/// see [`LobpcgOperator`]. Optional constraint vectors `Y` (`n × c`) exclude a
/// known subspace: all iterates are kept `B`-orthogonal to them.
///
/// Each iteration applies the Rayleigh–Ritz method to the span of the current
/// block `X`, the preconditioned residuals `T R` and the previous search
/// directions `P`, after `B`-orthonormalizing them against `Y` and each other
/// by classical Gram–Schmidt with reorthogonalization. The projected
/// eigenproblem has order at most `3k` for block size `k` and is solved with
/// [`HermitianEigenDecomposition`].
///
/// `A` and `B` are applied to each new column of `T R` and `P` after it is
/// orthonormalized, so the products carry no cancellation error from the
/// projection. An iteration therefore costs `2k` applications of `A`, `2k` of
/// `B` (when given) and `k` of the preconditioner. The products `A X` and
/// `B X` follow from the Rayleigh–Ritz update; they are recomputed after the
/// first iteration, every 50 iterations, at apparent convergence and at the
/// iteration limit, and only the residuals of these fresh products decide
/// acceptance.
///
/// The Hermitian assumption is checked on the projected matrix: the solve
/// fails when `vᵢᴴ A vⱼ` and `conj(vⱼᴴ A vᵢ)` differ by more than
/// `2 τ α ‖vᵢ‖₂ ‖vⱼ‖₂`, where `τ = max(√ε, n ε)` and
/// `α = maxₗ ‖A vₗ‖₂ / ‖vₗ‖₂` over the trial basis. The test is invariant
/// under scaling of `A` and independent of the residual tolerance; it detects
/// a skew-Hermitian part of `A` larger than about `τ ‖A‖₂` on the trial basis.
///
/// The value holds only the settings; `n` and `c` are read from the operator
/// and the constraints of each call. Defaults: smallest eigenvalues, at most
/// 100 000 operator applications, and residual tolerance `atol = rtol = √ε`.
/// The iteration limit of the execution options bounds the outer iterations
/// (default 100). The dense eigenproblems of the Rayleigh–Ritz steps use the
/// other execution options but no iteration limit.
///
/// # Examples
///
/// ```
/// use scioxide_core::BlockVectorView;
/// use scioxide_linalg::{LobpcgSolver, Matrix, SpectralTermination};
/// let a = Matrix::<f64>::identity(3)?;
/// let initial = BlockVectorView::new(&[1.0, 0.0, 0.0], 3, 1)?;
/// let constraints = BlockVectorView::new(&[], 3, 0)?;
/// let (output, report) =
///     LobpcgSolver::new(1).solve(&a.view().try_finite()?, initial, constraints)?;
/// assert_eq!(report.termination(), SpectralTermination::Converged);
/// // Every eigenvalue of the identity is 1.
/// assert!((output.eigenvalues()[0] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Knyazev, "Toward the optimal preconditioned eigensolver: locally
/// optimal block preconditioned conjugate gradient method",
/// *SIAM J. Sci. Comput.* 23(2), 2001.
#[derive(Debug, Clone, Copy)]
pub struct LobpcgSolver<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Requested block width.
    count: usize,
    /// Algebraic end of the spectrum: `Smallest` (default) or `Largest`.
    selection: LobpcgSelection,
    /// Provider-call budget, including mass/preconditioner and independent residual probes.
    applications: Budget,
    /// Original generalized residual acceptance.
    tolerance: scioxide_core::ResidualTolerance<R>,
}
impl<R: RealScalar> LobpcgSolver<R> {
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

    /// Sets which end of the spectrum to compute (default `Smallest`).
    pub const fn with_selection(mut self, selection: LobpcgSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Sets the maximum number of applications of `A`, `B` and the
    /// preconditioner, including residual checks (default 100 000).
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the residual tolerance (default `atol = rtol = √ε`).
    ///
    /// A Ritz pair `(λ, x)` is accepted when `‖A x − λ B x‖₂ ≤ atol + rtol ‖A x‖₂`.
    pub const fn with_tolerance(mut self, tolerance: scioxide_core::ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the block size, the number of eigenpairs computed.
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns which end of the spectrum is computed.
    pub const fn selection(&self) -> LobpcgSelection {
        self.selection
    }

    fn iteration_budget(&self) -> Budget {
        self.execution
            .iteration_budget()
            .unwrap_or(Budget::ONE.saturating_add(99))
    }

    /// Returns the maximum number of operator and preconditioner applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the residual tolerance.
    pub const fn tolerance(&self) -> scioxide_core::ResidualTolerance<R> {
        self.tolerance
    }
}
impl<R: RealScalar> LobpcgSolver<R> {
    /// Returns default options for `count` eigenpairs.
    pub fn new(count: usize) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            count,
            selection: LobpcgSelection::Smallest,
            applications: Budget::ONE.saturating_add(99_999),
            tolerance: scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
            ),
        }
    }
}
/// Eigenpairs computed by [`LobpcgSolver`]: Ritz values, Ritz vectors and residual norms.
///
/// Values are ordered from the selected end of the spectrum: ascending for
/// `Smallest`, descending for `Largest`. The vectors are the columns of an
/// `n × count` matrix, normalized so that `xᴴ B x = 1`. The residuals are
/// `‖A x − λ B x‖₂`.
///
/// After a limit or cancellation, the buffer holds the last block whose
/// residuals were recomputed from fresh products with `A` and `B`. If there is
/// none, the buffer is empty and [`LobpcgReport::iterations`] is zero.
#[derive(Debug)]
pub struct LobpcgBuffer<S: NumericScalar> {
    /// Real Ritz values, with requested ordering.
    pub(crate) eigenvalues: Vec<S::Real>,
    /// Ritz vectors as columns, normalized in the mass inner product.
    pub(crate) eigenvectors: Matrix<S>,
    /// Euclidean original residuals `||A x-lambda B x||`.
    pub(crate) residuals: Vec<S::Real>,
    count: usize,
}
impl<S: NumericScalar> LobpcgBuffer<S> {
    /// Allocates a buffer for `count` eigenpairs of `n × n` problems.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n · count` overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(n: usize, count: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            count: 0,
            eigenvalues: try_filled(
                count,
                S::Real::ZERO,
                scioxide_core::quantity!("result.eigenvalues"),
            )?,
            eigenvectors: Matrix::zeros(n, count)?,
            residuals: try_filled(
                count,
                S::Real::ZERO,
                scioxide_core::quantity!("result.residuals"),
            )?,
        })
    }
}
/// Workspace of [`LobpcgSolver`].
///
/// It holds O(n (c + k)) scalars for `c` constraints and block size `k`.
#[derive(Debug)]
pub struct LobpcgWorkspace<S: NumericScalar> {
    candidate_iteration: usize,
    slot: SolveSlot<()>,
    n: usize,
    k: usize,
    constraints: usize,
    capacity: usize,
    basis: Vec<S>,
    mass_basis: Vec<S>,
    images: Vec<S>,
    /// Euclidean norms of the trial-basis columns of the Hermitian test.
    lengths: Vec<S::Real>,
    x: Vec<S>,
    bx: Vec<S>,
    ax: Vec<S>,
    p: Vec<S>,
    residual: Vec<S>,
    vector: Vec<S>,
    mass_vector: Vec<S>,
    image: Vec<S>,
    coefficients: Vec<S>,
    provider: Vec<S>,
    mass_scratch: Vec<S>,
    preconditioner_scratch: Vec<S>,
    projected: Matrix<S>,
    eigen: HermitianEigenDecomposition<S>,
    eigen_work: LinalgWorkspace<'static, S>,
    candidate: LobpcgBuffer<S>,
    proposal: LobpcgBuffer<S>,
}
#[derive(Debug, Clone, Copy)]
struct LobpcgCall<R: RealScalar> {
    n: usize,
    constraints: usize,
    options: LobpcgSolver<R>,
}
impl<S: NumericScalar> ActionSlot for LobpcgWorkspace<S> {
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> LobpcgWorkspace<S> {
    /// Returns the attempt record of the most recent solve, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
impl<R: RealScalar> LobpcgCall<R> {
    fn workspace_requirements<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
    ) -> Result<LobpcgRequirements<S>, SciOxideError> {
        let op = op.into();
        self.validate(op)?;
        let LobpcgOperator {
            a,
            mass,
            preconditioner,
        } = op;
        let n = self.n;
        let k = self.options.count;
        let c = sum([self.constraints, product(&[3, k])?])?.min(n);
        let eigen = HermitianEigenDecomposition::<S>::workspace_requirements(
            &HermitianEigenWorkspaceRequest::new(
                c,
                &HermitianEigenOptions::<R>::default().with_execution_options(
                    super::projected_execution_options(self.options.execution),
                ),
            ),
        )?;
        let payload = sum([
            bytes::<S>(sum([
                c,
                product(&[3, n, c])?,
                product(&[7, n, k])?,
                product(&[3, n])?,
                product(&[2, c, c])?,
                a.apply_workspace_len()?,
                mass.map_or(Ok(0), |b| b.apply_workspace_len())?,
                preconditioner.map_or(Ok(0), |p| p.precondition_workspace_len())?,
            ])?)?,
            bytes::<R>(sum([product(&[2, c])?, product(&[4, k])?])?)?,
        ])?;
        LobpcgRequirements::new(
            LobpcgLayout {
                n: self.n,
                count: self.options.count,
                constraints: self.constraints,
                provider: a.apply_workspace_len()?,
                mass: mass.map_or(Ok(0), |b| b.apply_workspace_len())?,
                preconditioner: preconditioner.map_or(Ok(0), |p| p.precondition_workspace_len())?,
            },
            self.options.execution,
            payload,
            [eigen, Default::default()],
            Default::default(),
        )
    }
    fn workspace<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
    ) -> Result<LobpcgWorkspace<S>, SciOxideError> {
        let requirements = self.workspace_requirements(op)?;
        let mut workspace = LobpcgWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    fn solve<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        initial: BlockVectorView<'_, S>,
        constraints: BlockVectorView<'_, S>,
    ) -> Result<(LobpcgBuffer<S>, LobpcgReport), SciOxideError> {
        let op = op.into();
        let mut workspace = self.workspace(op)?;
        let mut output = LobpcgBuffer::new(self.n, self.options.count)?;
        let report = self.solve_into(
            op,
            initial,
            constraints,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn validate<S: NumericScalar<Real = R>>(
        &self,
        op: LobpcgOperator<'_, S>,
    ) -> Result<(), SciOxideError> {
        let LobpcgOperator {
            a,
            mass,
            preconditioner,
        } = op;
        let shape = [self.n, self.n];
        crate::checked::expect_shape(shape, [a.rows(), a.cols()])?;
        if let Some(b) = mass {
            crate::checked::expect_shape(shape, [b.rows(), b.cols()])?;
        }
        if let Some(p) = preconditioner {
            crate::checked::expect_shape([self.n], [p.dimension()])?;
        }
        Ok(())
    }
    fn solve_into<'o, 's, 'c, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        initial: BlockVectorView<'_, S>,
        constraints: BlockVectorView<'_, S>,
        result: &mut LobpcgBuffer<S>,
        w: &mut LobpcgWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LobpcgReport, SciOxideError> {
        w.slot.reject();
        let op = op.into();
        let requirements = self.workspace_requirements(op)?;
        let (n, k) = (self.n, self.options.count);
        crate::checked::expect_shape([n, k], [initial.rows(), initial.right_hand_sides()])?;
        crate::checked::expect_shape(
            [n, self.constraints],
            [constraints.rows(), constraints.right_hand_sides()],
        )?;
        crate::checked::expect_shape(
            [n, k],
            [result.eigenvectors.rows(), result.eigenvectors.cols()],
        )?;
        crate::checked::expect_shape([k], [result.eigenvalues.len()])?;
        crate::checked::expect_shape([k], [result.residuals.len()])?;
        for block in [initial, constraints] {
            crate::checked::validate_values(
                block.as_slice(),
                scioxide_core::NonFiniteOrigin::Input,
                "LOBPCG basis must be finite",
            )?;
        }
        w.prepare(&requirements)?;
        run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "LOBPCG", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| {
                self.run(
                    op,
                    LobpcgBasis {
                        initial,
                        constraints,
                    },
                    result,
                    w,
                    work,
                )
            },
            |report: &LobpcgReport| report.termination() == SpectralTermination::Converged,
        )
    }

    fn run<S: NumericScalar<Real = R>>(
        &self,
        providers: LobpcgOperator<'_, S>,
        basis: LobpcgBasis<'_, S>,
        result: &mut LobpcgBuffer<S>,
        w: &mut LobpcgWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<LobpcgReport, SciOxideError> {
        let LobpcgOperator {
            a,
            mass,
            preconditioner,
        } = providers;
        let LobpcgBasis {
            initial,
            constraints,
        } = basis;

        let execution = work.execution;

        let k = self.options.count;
        if w.provider.len() != a.apply_workspace_len()?
            || w.mass_scratch.len() != mass.map_or(Ok(0), |b| b.apply_workspace_len())?
            || w.preconditioner_scratch.len()
                != preconditioner.map_or(Ok(0), |p| p.precondition_workspace_len())?
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("op"),
                "provider changed its scratch requirement after preparation",
            ));
        }
        w.x.copy_from_slice(initial.as_slice());
        w.p.fill(S::ZERO);
        w.residual.fill(S::ZERO);
        w.candidate_iteration = 0;
        let mut termination = if k == 0 {
            SpectralTermination::Converged
        } else {
            SpectralTermination::IterationLimit
        };
        let outcome = (|| {
            execution.check()?;
            let mut excluded = 0;
            'iterations: for iteration in 0..if k == 0 {
                0
            } else {
                self.options.iteration_budget().get()
            } {
                if !work.iteration()? {
                    termination = SpectralTermination::EvaluationLimit;
                    break 'iterations;
                }
                let core::ops::ControlFlow::Continue(subspace) = self.build_trial_basis(
                    w,
                    providers,
                    constraints,
                    iteration,
                    &mut excluded,
                    &mut work,
                )?
                else {
                    termination = SpectralTermination::EvaluationLimit;
                    break 'iterations;
                };
                let converged = self.update_ritz_block(w, subspace, &mut work)?;
                if iteration == 0
                    || (iteration + 1).is_multiple_of(VERIFICATION_INTERVAL)
                    || iteration + 1 == self.options.iteration_budget().get()
                    || converged
                {
                    let core::ops::ControlFlow::Continue(verified) =
                        self.verify_block(a, mass, w, &mut work)?
                    else {
                        termination = SpectralTermination::EvaluationLimit;
                        break;
                    };
                    core::mem::swap(&mut w.candidate, &mut w.proposal);
                    w.candidate_iteration = iteration + 1;
                    if verified {
                        termination = SpectralTermination::Converged;
                        break;
                    }
                }
            }
            execution.check()
        })();
        termination = super::action_termination(outcome, termination)?;
        if w.candidate_iteration > 0 || k == 0 {
            result.eigenvalues.copy_from_slice(&w.candidate.eigenvalues);
            result.residuals.copy_from_slice(&w.candidate.residuals);
            result
                .eigenvectors
                .as_mut_slice()
                .copy_from_slice(w.candidate.eigenvectors.as_slice());
        }
        result.count = if w.candidate_iteration > 0 { k } else { 0 };
        if result.count == 0 {
            result.eigenvalues.fill(R::ZERO);
            result.residuals.fill(R::ZERO);
            result.eigenvectors.as_mut_slice().fill(S::ZERO);
        }
        Ok(LobpcgReport {
            termination,
            applications: work.count,
            iterations: w.candidate_iteration,
        })
    }

    fn build_trial_basis<S: NumericScalar<Real = R>>(
        &self,
        w: &mut LobpcgWorkspace<S>,
        providers: LobpcgOperator<'_, S>,
        constraints: BlockVectorView<'_, S>,
        iteration: usize,
        excluded_count: &mut usize,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<core::ops::ControlFlow<(), (usize, usize, usize)>, SciOxideError> {
        let n = self.n;
        let k = self.options.count;
        let LobpcgOperator {
            a,
            mass,
            preconditioner,
        } = providers;
        // Constraints and all their images are retained for the entire solve.
        let mut used = *excluded_count;
        if iteration == 0 {
            for column in 0..self.constraints {
                w.vector.copy_from_slice(constraints.column(column)?);
                if !insert(w, a, mass, &mut used, work)? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
            }
            *excluded_count = used;
        }
        let excluded = used;
        for column in 0..k {
            let range = column * n..(column + 1) * n;
            w.vector.copy_from_slice(&w.x[range.clone()]);
            if iteration == 0 {
                if !insert(w, a, mass, &mut used, work)? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
            } else {
                // The Ritz block is already B-orthonormal and B-orthogonal to
                // the constraints, so its cached products lose no accuracy.
                w.image.copy_from_slice(&w.ax[range.clone()]);
                w.mass_vector.copy_from_slice(&w.bx[range]);
                insert_cached(w, &mut used, work)?;
            }
        }
        if used != excluded + k {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("initial"),
                "LOBPCG initial block is dependent after constraints",
            ));
        }
        let old_end = used;
        if iteration > 0 {
            for column in 0..k {
                if let Some(p) = preconditioner {
                    if !work.precondition(
                        p,
                        &w.residual[column * n..(column + 1) * n],
                        &mut w.vector,
                        &mut w.preconditioner_scratch,
                    )? {
                        return Ok(core::ops::ControlFlow::Break(()));
                    }
                } else {
                    w.vector
                        .copy_from_slice(&w.residual[column * n..(column + 1) * n]);
                }
                if used < w.capacity && !insert(w, a, mass, &mut used, work)? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
            }
            for column in 0..k {
                w.vector.copy_from_slice(&w.p[column * n..(column + 1) * n]);
                if used < w.capacity && !insert(w, a, mass, &mut used, work)? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
            }
        }
        Ok(core::ops::ControlFlow::Continue((used, excluded, old_end)))
    }
    fn update_ritz_block<S: NumericScalar<Real = R>>(
        &self,
        w: &mut LobpcgWorkspace<S>,
        subspace: (usize, usize, usize),
        work: &mut Work<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = self.n;
        let k = self.options.count;
        let (used, excluded, old_end) = subspace;
        let dim = used - excluded;
        // Rayleigh-Ritz on `V^H A V`. Both `v_i^H A v_j` and
        // `conj(v_j^H A v_i)` are formed; a mismatch beyond the rounding
        // allowance shows that A is not Hermitian, otherwise their average is
        // used so the projected matrix is exactly Hermitian. The allowance is
        // relative to `alpha = max_l |A v_l| / |v_l|`, a lower bound of `|A|`,
        // because a column near the null space of A has an image made of
        // rounding errors of size `eps |A| |v|`.
        w.projected.try_resize_flat(dim, dim)?;
        let mut alpha = R::ZERO;
        for (l, length) in w.lengths[..dim].iter_mut().enumerate() {
            let column = (excluded + l) * n..(excluded + l + 1) * n;
            *length = finite::norm(&w.basis[column.clone()])?;
            if *length > R::ZERO {
                alpha = alpha.max(finite::norm(&w.images[column])? / *length);
            }
        }
        let epsilon = <R as RealScalar>::EPSILON;
        let allowance = epsilon
            .sqrt()
            .max(epsilon * crate::checked::rounded_count::<R>(n)?)
            * alpha;
        for i in 0..dim {
            for j in i..dim {
                let v = crate::reduction::dot_with_context(
                    VectorView::from_slice(&w.basis[(excluded + i) * n..(excluded + i + 1) * n]),
                    VectorView::from_slice(&w.images[(excluded + j) * n..(excluded + j + 1) * n]),
                    true,
                    work.kernels,
                )?;
                let opposite = crate::reduction::dot_with_context(
                    VectorView::from_slice(&w.basis[(excluded + j) * n..(excluded + j + 1) * n]),
                    VectorView::from_slice(&w.images[(excluded + i) * n..(excluded + i + 1) * n]),
                    true,
                    work.kernels,
                )?
                .conj();
                let bound = allowance * w.lengths[i] * w.lengths[j];
                if (v - opposite).abs() > bound + bound {
                    return Err(crate::checked::inconsistent(
                        scioxide_core::argument!("a"),
                        "LOBPCG operator violates Hermitian assumption",
                    ));
                }
                let average = (v + opposite) * S::from_real(R::HALF);
                w.projected.as_mut_slice()[i * dim + j] = average;
                w.projected.as_mut_slice()[j * dim + i] = average.conj();
            }
        }
        HermitianEigenDecomposition::decompose_into(
            &w.projected,
            crate::HermitianInput::Exact,
            &HermitianEigenOptions::default()
                .with_execution_options(super::projected_execution_options(self.options.execution)),
            &mut w.eigen,
            &mut w.eigen_work,
            work.kernels.scoped(&mut *work.solve),
        )?;
        let mut converged = true;
        for column in 0..k {
            let index = if self.options.selection == LobpcgSelection::Largest {
                dim - 1 - column
            } else {
                column
            };
            // Ritz vector `x = V y`; the new direction `p` is the part of
            // `V y` coming from the non-X columns (T R and old P), which
            // is the implicit, cancellation-free form of `x_new - x_old`.
            // Its products are formed when it enters the next trial basis.
            let lambda = w.eigen.eigenvalues()[index];
            let coefficients = || (0..dim).map(|j| w.eigen.eigenvectors().get_element(j, index));
            let range = column * n..(column + 1) * n;
            subspace::combine(
                &mut w.x[range.clone()],
                &w.basis[excluded * n..used * n],
                coefficients(),
                core::convert::identity,
                work.kernels.dispatch(),
            )?;
            subspace::combine(
                &mut w.ax[range.clone()],
                &w.images[excluded * n..used * n],
                coefficients(),
                core::convert::identity,
                work.kernels.dispatch(),
            )?;
            subspace::combine(
                &mut w.bx[range.clone()],
                &w.mass_basis[excluded * n..used * n],
                coefficients(),
                core::convert::identity,
                work.kernels.dispatch(),
            )?;
            subspace::combine(
                &mut w.p[range.clone()],
                &w.basis[old_end * n..used * n],
                coefficients().skip(old_end - excluded),
                core::convert::identity,
                work.kernels.dispatch(),
            )?;
            for row in 0..n {
                w.proposal.eigenvectors.as_mut_slice()[row * k + column] = w.x[column * n + row];
            }
            let anorm = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.ax[range.clone()]),
                work.kernels,
            )?;
            for ((r, &ax), &bx) in w.residual[range.clone()]
                .iter_mut()
                .zip(&w.ax[range.clone()])
                .zip(&w.bx[range.clone()])
            {
                *r = crate::checked::sub(ax, S::from_real(lambda) * bx)?;
            }
            let r = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.residual[range]),
                work.kernels,
            )?;
            w.proposal.eigenvalues[column] = lambda;
            w.proposal.residuals[column] = r;
            converged &= r <= self.options.tolerance.threshold(anorm)?;
        }
        Ok(converged)
    }
    /// Refresh the recurrence images and accept only a completely checked block.
    fn verify_block<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn LinearOperator<S>,
        mass: Option<&dyn LinearOperator<S>>,
        w: &mut LobpcgWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<core::ops::ControlFlow<(), bool>, SciOxideError> {
        let mut converged = true;
        for column in 0..w.k {
            let range = column * w.n..(column + 1) * w.n;
            if !work.apply(
                a,
                &w.x[range.clone()],
                &mut w.ax[range.clone()],
                &mut w.provider,
            )? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            if let Some(b) = mass {
                if !work.apply(
                    b,
                    &w.x[range.clone()],
                    &mut w.bx[range.clone()],
                    &mut w.mass_scratch,
                )? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
            } else {
                w.bx[range.clone()].copy_from_slice(&w.x[range.clone()]);
            }
            let scale = finite::norm(&w.ax[range.clone()])?;
            let lambda = S::from_real(w.proposal.eigenvalues[column]);
            for ((r, &a), &b) in w.residual[range.clone()]
                .iter_mut()
                .zip(&w.ax[range.clone()])
                .zip(&w.bx[range.clone()])
            {
                *r = crate::checked::sub(a, lambda * b)?;
            }
            let residual = finite::norm(&w.residual[range])?;
            w.proposal.residuals[column] = residual;
            converged &= residual <= self.options.tolerance.threshold(scale)?;
        }
        Ok(core::ops::ControlFlow::Continue(converged))
    }
}
impl<R: RealScalar> LobpcgSolver<R> {
    /// Checks the settings against the order `n` and `constraints` constraint
    /// vectors and returns the per-call value.
    fn call(&self, n: usize, constraints: usize) -> Result<LobpcgCall<R>, SciOxideError> {
        if checked_add(
            constraints,
            self.count,
            scioxide_core::quantity!("constrained_columns"),
        )? > n
        {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("self.count"),
                "LOBPCG block exceeds unconstrained dimension",
            ));
        }
        Ok(LobpcgCall {
            n,
            constraints,
            options: *self,
        })
    }

    /// Returns the order of the square operator `A` of `op`.
    fn order<S: NumericScalar<Real = R>>(
        op: LobpcgOperator<'_, S>,
    ) -> Result<usize, SciOxideError> {
        crate::checked::square(op.a.rows(), op.a.cols())?;
        Ok(op.a.rows())
    }
    /// Returns the workspace storage needed to solve the problem `op` with
    /// `constraints` constraint vectors.
    ///
    /// No operator is applied; only shapes and scratch lengths are read. The
    /// storage includes the dense projected eigensolver. The settings are
    /// checked first, so the execution methods report the same errors before
    /// any work.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `A` is not square, `B` is not `n × n`, or the
    ///   preconditioner dimension is not `n`.
    /// - `InvalidInput` (`OutOfRange`, argument `self.count`) if the block size
    ///   plus the number of constraints exceeds `n`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of the operators.
    pub fn workspace_requirements<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        constraints: usize,
    ) -> Result<LobpcgRequirements<S>, SciOxideError> {
        let op = op.into();
        self.call(Self::order(op)?, constraints)?
            .workspace_requirements(op)
    }

    /// Allocates a workspace for `op`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        constraints: usize,
    ) -> Result<LobpcgWorkspace<S>, SciOxideError> {
        let op = op.into();
        self.call(Self::order(op)?, constraints)?.workspace(op)
    }

    /// Computes the eigenpairs with a new workspace and result buffer.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`LobpcgBuffer::new`] and
    /// [`Self::solve_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::BlockVectorView;
    /// use scioxide_linalg::{LobpcgSolver, Matrix};
    /// let a = Matrix::<f64>::identity(3)?;
    /// let method = LobpcgSolver::new(1);
    /// let (output, report) = method.solve(
    ///     &a.view().try_finite()?,
    ///     BlockVectorView::new(&[1., 0., 0.], 3, 1)?,
    ///     BlockVectorView::new(&[], 3, 0)?,
    /// )?;
    /// // Every eigenvalue of the identity is 1.
    /// assert!((output.eigenvalues()[0] - 1.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        initial: BlockVectorView<'_, S>,
        constraints: BlockVectorView<'_, S>,
    ) -> Result<(LobpcgBuffer<S>, LobpcgReport), SciOxideError> {
        let op = op.into();
        self.call(Self::order(op)?, constraints.right_hand_sides())?
            .solve(op, initial, constraints)
    }

    /// Computes eigenpairs of `op` from the starting block `initial` and writes them to `result`.
    ///
    /// `initial` is `n × count` and `constraints` is `n × c`, each stored one
    /// column after another as in [`BlockVectorView`]. Numerically dependent
    /// constraint columns are dropped. The initial block must stay linearly
    /// independent after it is made `B`-orthogonal to the constraints.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `initial`, `constraints` or `result` has the
    ///   wrong shape.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `NonFinite` if `initial` or `constraints` has a non-finite entry, or
    ///   if an operator or the preconditioner returns a non-finite value.
    /// - `InvalidInput` (`Inconsistent`) if the initial block becomes linearly
    ///   dependent, if the projected matrix shows that `A` is not Hermitian, or
    ///   if a scratch length changed after the workspace was reserved.
    /// - `DomainError` if `B` is not positive definite on the search subspace.
    /// - `NumericalInstability` (`Overflow`) if a residual overflows.
    /// - Errors of the dense Hermitian eigensolver, and errors returned by the
    ///   operators or the preconditioner.
    pub fn solve_into<'o, 's, 'c, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<LobpcgOperator<'o, S>>,
        initial: BlockVectorView<'_, S>,
        constraints: BlockVectorView<'_, S>,
        result: &mut LobpcgBuffer<S>,
        w: &mut LobpcgWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LobpcgReport, SciOxideError> {
        let op = op.into();
        self.call(Self::order(op)?, constraints.right_hand_sides())?
            .solve_into(op, initial, constraints, result, w, context)
    }
}

/// Appends the column in `w.vector` to the trial basis with fresh products.
///
/// The column is scaled to unit Euclidean norm and `B`-orthogonalized against
/// the stored basis `Q` by classical Gram–Schmidt applied twice, with the
/// coefficients `(B Q)ᴴ v` taken from the stored `B` images. `B` and then `A`
/// are applied to the final normalized column, so its products carry only the
/// rounding of one application however much the projection cancelled. A
/// column that is numerically dependent on the basis is skipped before `A` is
/// applied. Returns `false` when the application budget is exhausted.
fn insert<S: NumericScalar>(
    w: &mut LobpcgWorkspace<S>,
    a: &dyn LinearOperator<S>,
    mass: Option<&dyn LinearOperator<S>>,
    used: &mut usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<bool, SciOxideError> {
    let n = w.n;
    let euclidean =
        crate::reduction::l2_norm_with_context(VectorView::from_slice(&w.vector), work.kernels)?;
    if euclidean == S::Real::ZERO {
        return Ok(true);
    }
    for value in &mut w.vector {
        *value = crate::checked::finite_candidate(*value / S::from_real(euclidean))?;
    }
    let dispatch = work.kernels.dispatch();
    let basis = Columns::Packed(&w.basis[..*used * n], n);
    let mass_basis = Columns::Packed(&w.mass_basis[..*used * n], n);
    let basis_maximum = basis.maximum(dispatch)?;
    let mass_maximum = mass_basis.maximum(dispatch)?;
    let coefficients = &mut w.coefficients[..*used];
    // `B`-norm² of the removed component; with that of the remainder it gives
    // the `B`-norm² of the scaled input by the Pythagorean theorem.
    let mut removed = S::Real::ZERO;
    for _ in 0..2 {
        level2::project_columns(
            mass_basis,
            mass_maximum,
            &w.vector,
            finite::maximum(&w.vector, dispatch)?,
            coefficients,
            dispatch,
        )?;
        level2::subtract_columns(basis, basis_maximum, coefficients, &mut w.vector)?;
        removed = coefficients
            .iter()
            .fold(removed, |sum, &c| sum + c.abs() * c.abs());
    }
    if let Some(b) = mass {
        if !work.apply(b, &w.vector, &mut w.mass_vector, &mut w.mass_scratch)? {
            return Ok(false);
        }
    } else {
        w.mass_vector.copy_from_slice(&w.vector);
    }
    let value = crate::reduction::dot_with_context(
        VectorView::from_slice(&w.vector),
        VectorView::from_slice(&w.mass_vector),
        true,
        work.kernels,
    )?;
    let Some(norm) = mass_norm(value, removed + value.abs())? else {
        return Ok(true);
    };
    let column = *used * n..(*used + 1) * n;
    for ((q, m), (&v, &bv)) in w.basis[column.clone()]
        .iter_mut()
        .zip(&mut w.mass_basis[column.clone()])
        .zip(w.vector.iter().zip(&w.mass_vector))
    {
        *q = v / S::from_real(norm);
        *m = bv / S::from_real(norm);
    }
    if !work.apply(
        a,
        &w.basis[column.clone()],
        &mut w.images[column],
        &mut w.provider,
    )? {
        return Ok(false);
    }
    *used += 1;
    Ok(true)
}

/// Appends a Ritz vector in `w.vector` with its cached products `B x` in
/// `w.mass_vector` and `A x` in `w.image`.
///
/// The Ritz block is `B`-orthonormal and `B`-orthogonal to the constraints, so
/// the projection removes only rounding errors and the products are updated
/// with it at no loss of accuracy. A dependent vector is skipped.
fn insert_cached<S: NumericScalar>(
    w: &mut LobpcgWorkspace<S>,
    used: &mut usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let n = w.n;
    // Remove an extreme Euclidean scale before forming B-inner products, so
    // that no squared norm underflows or overflows.
    let euclidean =
        crate::reduction::l2_norm_with_context(VectorView::from_slice(&w.vector), work.kernels)?;
    if euclidean != S::Real::ZERO && !(euclidean * euclidean).is_normal() {
        for values in [&mut w.vector, &mut w.mass_vector, &mut w.image] {
            for value in values {
                *value = crate::checked::finite_candidate(*value / S::from_real(euclidean))?;
            }
        }
    }
    let original = crate::reduction::dot_with_context(
        VectorView::from_slice(&w.vector),
        VectorView::from_slice(&w.mass_vector),
        true,
        work.kernels,
    )?
    .real_component();
    subspace::orthogonalize(
        &mut w.vector,
        Columns::Packed(&w.basis[..*used * n], n),
        &mut w.coefficients,
        Some((
            &mut w.mass_vector,
            Columns::Packed(&w.mass_basis[..*used * n], n),
        )),
        Some((&mut w.image, Columns::Packed(&w.images[..*used * n], n))),
        |_, _| {},
        work.kernels,
    )?;
    let value = crate::reduction::dot_with_context(
        VectorView::from_slice(&w.vector),
        VectorView::from_slice(&w.mass_vector),
        true,
        work.kernels,
    )?;
    let Some(norm) = mass_norm(value, original.abs())? else {
        return Ok(());
    };
    for i in 0..n {
        w.basis[*used * n + i] = w.vector[i] / S::from_real(norm);
        w.mass_basis[*used * n + i] = w.mass_vector[i] / S::from_real(norm);
        w.images[*used * n + i] = w.image[i] / S::from_real(norm);
    }
    *used += 1;
    Ok(())
}

/// Returns the `B`-norm `√value` of a projected column whose input had
/// squared `B`-norm `original`, or `None` when the column is numerically
/// dependent (`value ≤ ε² original`).
///
/// # Errors
///
/// `DomainError` if `value` shows that `B` is not Hermitian positive definite
/// on the trial subspace.
fn mass_norm<S: NumericScalar>(
    value: S,
    original: S::Real,
) -> Result<Option<S::Real>, SciOxideError> {
    let epsilon = <S::Real as RealScalar>::EPSILON;
    if value.real_component() < -epsilon * original
        || value.imaginary_component().abs() > epsilon.sqrt() * value.abs()
    {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("mass"),
            scioxide_core::DomainReason::OutsideDomain,
            "mass operator is not positive definite Hermitian on the trial subspace",
        ));
    }
    if value.real_component() <= epsilon * epsilon * original {
        return Ok(None);
    }
    Ok(Some(value.real_component().sqrt()))
}

/// Operators of a LOBPCG solve: Hermitian `A`, optional mass `B` and optional preconditioner.
///
/// A reference to a sized operator converts into the standard problem
/// `A x = λ x` without a preconditioner.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{LobpcgOperator, Matrix};
/// let a = Matrix::<f64>::identity(3)?;
/// let a = a.view().try_finite()?;
/// let b = Matrix::<f64>::identity(3)?;
/// let b = b.view().try_finite()?;
/// let pencil = LobpcgOperator::new(&a).with_mass(&b);
/// assert_eq!(format!("{pencil:?}"), "LobpcgOperator { mass: true, preconditioner: false, .. }");
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct LobpcgOperator<'a, S: NumericScalar> {
    a: &'a dyn LinearOperator<S>,
    mass: Option<&'a dyn LinearOperator<S>>,
    preconditioner: Option<&'a dyn Preconditioner<S>>,
}
impl<'a, S: NumericScalar> LobpcgOperator<'a, S> {
    /// Returns the standard problem `A x = λ x` without a preconditioner.
    pub fn new(a: &'a dyn LinearOperator<S>) -> Self {
        Self {
            a,
            mass: None,
            preconditioner: None,
        }
    }
    /// Sets the mass matrix `B` of the problem `A x = λ B x`.
    ///
    /// `B` must be Hermitian positive definite.
    #[must_use]
    pub fn with_mass(mut self, mass: &'a dyn LinearOperator<S>) -> Self {
        self.mass = Some(mass);
        self
    }
    /// Sets the preconditioner applied to the residuals.
    #[must_use]
    pub fn with_preconditioner(mut self, preconditioner: &'a dyn Preconditioner<S>) -> Self {
        self.preconditioner = Some(preconditioner);
        self
    }
}
impl<S: NumericScalar> core::fmt::Debug for LobpcgOperator<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LobpcgOperator")
            .field("mass", &self.mass.is_some())
            .field("preconditioner", &self.preconditioner.is_some())
            .finish_non_exhaustive()
    }
}
impl<'a, S: NumericScalar, T: LinearOperator<S>> From<&'a T> for LobpcgOperator<'a, S> {
    fn from(a: &'a T) -> Self {
        Self::new(a)
    }
}
struct LobpcgBasis<'a, S: NumericScalar> {
    initial: BlockVectorView<'a, S>,
    constraints: BlockVectorView<'a, S>,
}

impl<S: NumericScalar> LobpcgBuffer<S> {
    const fn empty() -> Self {
        Self {
            count: 0,
            eigenvalues: Vec::new(),
            eigenvectors: Matrix::empty(),
            residuals: Vec::new(),
        }
    }
}

impl<S: NumericScalar> StorageBytes for LobpcgBuffer<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.eigenvalues.storage_bytes()?,
            self.eigenvectors.storage_bytes()?,
            self.residuals.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> Default for LobpcgWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> LobpcgWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            candidate_iteration: 0,
            slot: SolveSlot::new(),
            n: 0,
            k: 0,
            constraints: 0,
            capacity: 0,
            basis: Vec::new(),
            mass_basis: Vec::new(),
            images: Vec::new(),
            lengths: Vec::new(),
            x: Vec::new(),
            bx: Vec::new(),
            ax: Vec::new(),
            p: Vec::new(),
            residual: Vec::new(),
            vector: Vec::new(),
            mass_vector: Vec::new(),
            image: Vec::new(),
            coefficients: Vec::new(),
            provider: Vec::new(),
            mass_scratch: Vec::new(),
            preconditioner_scratch: Vec::new(),
            projected: Matrix::empty(),
            eigen: HermitianEigenDecomposition::empty(),
            eigen_work: LinalgWorkspace::new(),
            candidate: LobpcgBuffer::empty(),
            proposal: LobpcgBuffer::empty(),
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
        requirements: &LobpcgRequirements<S>,
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
        requirements: &LobpcgRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for LobpcgWorkspace<S> {
    type Requirements = LobpcgRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &LobpcgRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let LobpcgLayout {
            n,
            count: k,
            constraints,
            provider,
            mass,
            preconditioner,
        } = requirements.layout;
        let capacity = sum([constraints, product(&[3, k])?])?.min(n);
        let size = product(&[n, capacity])?;
        let block = product(&[n, k])?;
        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.basis, size)?;
        plan.buffer(&self.mass_basis, size)?;
        plan.buffer(&self.images, size)?;
        plan.buffer(&self.lengths, capacity)?;
        plan.buffer(&self.x, block)?;
        plan.buffer(&self.bx, block)?;
        plan.buffer(&self.ax, block)?;
        plan.buffer(&self.p, block)?;
        plan.buffer(&self.residual, block)?;
        plan.buffer(&self.vector, n)?;
        plan.buffer(&self.mass_vector, n)?;
        plan.buffer(&self.coefficients, capacity)?;
        plan.buffer(&self.image, n)?;
        plan.buffer(&self.provider, provider)?;
        plan.buffer(&self.mass_scratch, mass)?;
        plan.buffer(&self.preconditioner_scratch, preconditioner)?;
        plan.matrix(&self.projected, capacity, capacity)?;
        self.eigen.action_storage(&mut plan, capacity)?;
        plan.dense(&self.eigen_work, &requirements.storage.dense[0])?;
        plan.matrix(&self.candidate.eigenvectors, n, k)?;
        plan.buffer(&self.candidate.eigenvalues, k)?;
        plan.buffer(&self.candidate.residuals, k)?;
        plan.matrix(&self.proposal.eigenvectors, n, k)?;
        plan.buffer(&self.proposal.eigenvalues, k)?;
        plan.buffer(&self.proposal.residuals, k)?;
        Ok(plan)
    }
    fn configure(&mut self, requirements: &LobpcgRequirements<S>) -> Result<(), SciOxideError> {
        let LobpcgLayout {
            n,
            count: k,
            constraints,
            provider,
            mass,
            preconditioner,
        } = requirements.layout;
        let capacity = sum([constraints, product(&[3, k])?])?.min(n);
        let size = product(&[n, capacity])?;
        let block = product(&[n, k])?;
        resize(&mut self.basis, size, S::ZERO)?;
        resize(&mut self.mass_basis, size, S::ZERO)?;
        resize(&mut self.images, size, S::ZERO)?;
        resize(&mut self.lengths, capacity, S::Real::ZERO)?;
        resize(&mut self.x, block, S::ZERO)?;
        resize(&mut self.bx, block, S::ZERO)?;
        resize(&mut self.ax, block, S::ZERO)?;
        resize(&mut self.p, block, S::ZERO)?;
        resize(&mut self.residual, block, S::ZERO)?;
        resize(&mut self.vector, n, S::ZERO)?;
        resize(&mut self.mass_vector, n, S::ZERO)?;
        resize(&mut self.coefficients, capacity, S::ZERO)?;
        resize(&mut self.image, n, S::ZERO)?;
        resize(&mut self.provider, provider, S::ZERO)?;
        resize(&mut self.mass_scratch, mass, S::ZERO)?;
        resize(&mut self.preconditioner_scratch, preconditioner, S::ZERO)?;
        self.projected.try_resize_flat(capacity, capacity)?;
        self.eigen.reserve(capacity, capacity)?;
        self.eigen_work
            .try_reserve(&requirements.storage.dense[0])?;
        self.candidate.eigenvectors.try_resize_flat(n, k)?;
        resize(&mut self.candidate.eigenvalues, k, S::Real::ZERO)?;
        resize(&mut self.candidate.residuals, k, S::Real::ZERO)?;
        self.proposal.eigenvectors.try_resize_flat(n, k)?;
        resize(&mut self.proposal.eigenvalues, k, S::Real::ZERO)?;
        resize(&mut self.proposal.residuals, k, S::Real::ZERO)?;
        self.n = n;
        self.k = k;
        self.constraints = constraints;
        self.capacity = capacity;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for LobpcgWorkspace<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.basis.storage_bytes()?,
            self.mass_basis.storage_bytes()?,
            self.images.storage_bytes()?,
            self.lengths.storage_bytes()?,
            self.x.storage_bytes()?,
            self.bx.storage_bytes()?,
            self.ax.storage_bytes()?,
            self.p.storage_bytes()?,
            self.residual.storage_bytes()?,
            self.vector.storage_bytes()?,
            self.mass_vector.storage_bytes()?,
            self.image.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.provider.storage_bytes()?,
            self.mass_scratch.storage_bytes()?,
            self.preconditioner_scratch.storage_bytes()?,
            self.projected.storage_bytes()?,
            self.eigen.storage_bytes()?,
            self.eigen_work.storage_bytes()?,
            self.candidate.storage_bytes()?,
            self.proposal.storage_bytes()?,
        ])
    }
}

/// Termination status and work counts of a LOBPCG solve.
#[derive(Debug, Clone, Copy)]
pub struct LobpcgReport {
    termination: SpectralTermination,
    applications: usize,
    iterations: usize,
}
impl LobpcgReport {
    /// Returns the termination status.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }
    /// Returns the number of applications of `A`, `B` and the preconditioner.
    pub const fn applications(&self) -> usize {
        self.applications
    }
    /// Returns the iteration of the block in the result buffer, or zero if it is empty.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
}

impl<S: NumericScalar> LobpcgBuffer<S> {
    /// Returns the Ritz values, ordered from the selected end of the spectrum.
    pub fn eigenvalues(&self) -> &[S::Real] {
        &self.eigenvalues[..self.count]
    }
    /// Returns the Ritz vectors as the columns of an `n × count` view, with `xᴴ B x = 1`.
    pub fn eigenvectors(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.eigenvectors.view();
        view.cols = self.count;
        view
    }
    /// Returns the residual norms `‖A x − λ B x‖₂`.
    pub fn residuals(&self) -> &[S::Real] {
        &self.residuals[..self.count]
    }
}
