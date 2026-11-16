//! Randomized low-rank SVD and column interpolative decomposition from a given sketch.
use super::workspace::{RandomizedLayout, RandomizedLowRankRequirements};
use super::{ActionSlot, SpectralTermination, Work, run_action};
use crate::kernel::subspace;
use crate::workspace::SolveSlot;
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};
use crate::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, MatrixView,
    RankPolicy, ThinSvdDecomposition, VectorView,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AdjointOperator, AttemptRecord, Budget, NumericScalar, RealScalar, SciOxideError, Tolerance,
};
/// Randomized low-rank SVD or interpolative decomposition from a sketch you provide.
///
/// `A` is an `m × n` operator, used only through products `A x` and `Aᴴ x`.
/// The method generates no random numbers. You pass the `n × w` test matrix `Ω`
/// (the sketch, whose columns are the test vectors; usually with independent
/// standard normal entries) and at least one separate probe vector. For the
/// same inputs the result is the same.
///
/// The range finder of Halko, Martinsson and Tropp (2011) computes an
/// orthonormal basis `Q` of the range of `A Ω` by Gram–Schmidt and drops the
/// vectors that the rank policy considers dependent. Each of the `q` power
/// steps replaces `Q` by an orthonormal basis of `A (Aᴴ Q)`. The small matrix
/// `Qᴴ A`, of size at most `w × n`, is formed from adjoint products.
/// [`Self::svd_into`] computes its SVD `Qᴴ A = Ũ Σ Vᴴ` and returns `U = Q Ũ`.
/// [`Self::interpolative_into`] computes its column-pivoted QR and selects `r`
/// columns of `A`. The cost is at most `w (2q + 2)` products, plus one per
/// probe, plus `r` for the interpolative decomposition.
///
/// The rank `r` of the result is the smallest of the target rank, the rank of
/// `Q`, and the numerical rank of the small factorization under the rank
/// policy. Each probe `g` (a column of `probes`) then gives the residual
/// `‖A g − Ã g‖₂`, which decides the termination status.
///
/// The value holds only the settings; `m`, `n` and `w` are read from the
/// operator and the sketch of each call. Defaults: rank policy
/// [`RankPolicy::Exact`], no power steps, at most 100 000 operator
/// applications, and probe tolerance `atol = rtol = √ε`. The rank policy
/// decides which vectors of the range basis are kept, by comparing the norm
/// left after orthogonalization with the norm before. It also truncates the
/// small SVD or pivoted QR.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RandomizedLowRank, SpectralTermination};
/// let a = Matrix::from_rows(&[[2.0_f64, 0.0], [0.0, 0.0]])?;
/// let sketch = Matrix::from_rows(&[[1.0], [1.0]])?;
/// let probes = Matrix::from_rows(&[[1.0], [0.0]])?;
/// let (output, report) =
///     RandomizedLowRank::new(1).svd(&a.view().try_finite()?, sketch.view(), probes.view())?;
/// assert_eq!(report.termination(), SpectralTermination::Converged);
/// // A = diag(2, 0) has the single nonzero singular value 2.
/// assert!((output.singular_values()[0] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// N. Halko, P.-G. Martinsson and J. A. Tropp, "Finding structure with
/// randomness: probabilistic algorithms for constructing approximate matrix
/// decompositions", *SIAM Review* 53(2), 2011.
#[derive(Debug, Clone, Copy)]
pub struct RandomizedLowRank<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Requested output rank, bounded by sketch width and matrix dimensions.
    rank: usize,
    rank_policy: RankPolicy<R>,
    /// Explicit normalized subspace iterations; zero uses the supplied sketch once.
    power_steps: usize,
    /// Maximum forward and adjoint provider applications.
    applications: Budget,
    /// Held-out probe residual acceptance; probes are caller supplied.
    tolerance: scioxide_core::ResidualTolerance<R>,
}
impl<R: RealScalar> RandomizedLowRank<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    ///
    /// Every sketch column and every power step counts as one iteration toward
    /// the iteration limit.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the rank policy for the range basis and the truncation of the small
    /// SVD or QR (default `Exact`).
    pub const fn with_rank_policy(mut self, policy: RankPolicy<R>) -> Self {
        self.rank_policy = policy;
        self
    }

    /// Returns the rank policy.
    pub const fn rank_policy(&self) -> RankPolicy<R> {
        self.rank_policy
    }

    /// Sets the number `q` of power iterations (default 0).
    ///
    /// Each step replaces the range basis `Q` by an orthonormal basis of
    /// `A (Aᴴ Q)`, orthonormalizing after each product. Power steps improve the
    /// basis when the singular values decay slowly, at the cost of about `2w`
    /// more products for sketch width `w`.
    pub const fn with_power_steps(mut self, power_steps: usize) -> Self {
        self.power_steps = power_steps;
        self
    }

    /// Sets the maximum number of forward plus adjoint operator applications
    /// (default 100 000).
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the probe tolerance (default `atol = rtol = √ε`).
    ///
    /// The result is accepted when `‖A g − Ã g‖₂ ≤ atol + rtol ‖A g‖₂` for every
    /// probe vector `g`, where `Ã` is the low-rank approximation.
    pub const fn with_tolerance(mut self, tolerance: scioxide_core::ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the target rank.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the number of power iterations.
    pub const fn power_steps(&self) -> usize {
        self.power_steps
    }

    /// Returns the maximum number of operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the probe tolerance.
    pub const fn tolerance(&self) -> scioxide_core::ResidualTolerance<R> {
        self.tolerance
    }
}
impl<R: RealScalar> RandomizedLowRank<R> {
    /// Returns default options for target rank `rank`.
    pub fn new(rank: usize) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            rank,
            rank_policy: RankPolicy::Exact,
            power_steps: 0,
            applications: Budget::ONE.saturating_add(99_999),
            tolerance: scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
            ),
        }
    }
}
/// Termination status, rank and probe residuals of a randomized low-rank approximation.
///
/// The status is `Converged` when every probe residual meets the tolerance and
/// `AccuracyLimit` when one does not. The probes are independent of the
/// sketch, so their residuals are an independent check of the approximation.
/// They are not a bound on its error.
///
/// After a limit or cancellation, the output holds the approximation if its
/// construction finished, and is zero with rank zero otherwise. The residuals
/// then cover only [`Self::completed_probes`] probes. With no completed probe
/// the accuracy is unknown, although the residual fields are zero.
#[derive(Debug, Clone, Copy)]
pub struct LowRankReport<R: RealScalar> {
    /// Probe acceptance, not an assertion of optimal approximation or a certificate.
    termination: SpectralTermination,
    /// Numerical rank of the returned approximation.
    rank: usize,
    /// Maximum absolute held-out probe residual.
    probe_residual: R,
    /// Maximum measured relative residual; absent for no probes or all-zero images.
    relative_probe_residual: Option<R>,
    /// Completed held-out probes; zero means no residual has been measured.
    completed_probes: usize,
    /// Actual forward plus adjoint provider applications.
    applications: usize,
}
/// Approximate truncated SVD `A ≈ U Σ Vᴴ` computed by [`RandomizedLowRank`].
///
/// The buffer has room for the target rank. The accessors return the first
/// `r` components, where `r` is the rank of the result.
#[derive(Debug)]
pub struct RandomizedSvdBuffer<S: NumericScalar> {
    /// First report.rank() columns contain left singular vectors.
    pub(crate) u: Matrix<S>,
    /// First report.rank() entries contain singular values.
    pub(crate) singular_values: Vec<S::Real>,
    /// First report.rank() rows contain conjugate-transposed right singular vectors.
    pub(crate) vt: Matrix<S>,
    rank: usize,
}
/// Column interpolative decomposition `A ≈ C Z` computed by [`RandomizedLowRank`].
///
/// `C` holds `r` columns of `A`, with indices [`Self::column_indices`], and `Z`
/// is `r × n`. Column `column_indices()[j]` of `Z` is the unit vector `eⱼ`, so
/// the selected columns of `A` are reproduced exactly.
#[derive(Debug)]
pub struct InterpolativeBuffer<S: NumericScalar> {
    /// Selected original columns occupy the first report.rank() columns.
    pub(crate) columns: Matrix<S>,
    /// First report.rank() rows contain coefficients in original column order.
    pub(crate) coefficients: Matrix<S>,
    /// Original column indices; use the first report.rank() entries.
    pub(crate) column_indices: Vec<usize>,
    rank: usize,
}
fn empty_report<R: RealScalar>() -> LowRankReport<R> {
    LowRankReport {
        termination: SpectralTermination::EvaluationLimit,
        rank: 0,
        probe_residual: R::ZERO,
        relative_probe_residual: None,
        applications: 0,
        completed_probes: 0,
    }
}
impl<S: NumericScalar> RandomizedSvdBuffer<S> {
    /// Allocates a buffer for rank-`rank` approximations of `m × n` operators.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(m: usize, n: usize, rank: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            u: Matrix::zeros(m, rank)?,
            singular_values: try_filled(
                rank,
                S::Real::ZERO,
                scioxide_core::quantity!("result.singular_values"),
            )?,
            vt: Matrix::zeros(rank, n)?,
            rank: 0,
        })
    }
}
impl<S: NumericScalar> InterpolativeBuffer<S> {
    /// Allocates a buffer for rank-`rank` decompositions of `m × n` operators.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(m: usize, n: usize, rank: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            columns: Matrix::zeros(m, rank)?,
            coefficients: Matrix::zeros(rank, n)?,
            column_indices: try_filled(rank, 0, scioxide_core::quantity!("result.column_indices"))?,
            rank: 0,
        })
    }
}
#[derive(Debug, Clone, Copy)]
struct RandomizedCall<R: RealScalar> {
    m: usize,
    n: usize,
    width: usize,
    options: RandomizedLowRank<R>,
}
/// Workspace of [`RandomizedLowRank`].
///
/// It holds O((m + n) w) scalars for sketch width `w`, plus the dense SVD and
/// QR of the `w × n` matrix `Qᴴ A`.
#[derive(Debug)]
pub struct RandomizedLowRankWorkspace<S: NumericScalar> {
    slot: SolveSlot<()>,
    m: usize,
    n: usize,
    width: usize,
    q: Vec<S>,
    z: Vec<S>,
    left: Vec<S>,
    coefficients: Vec<S>,
    right: Vec<S>,
    approximation: Vec<S>,
    forward: Vec<S>,
    adjoint: Vec<S>,
    small: Matrix<S>,
    svd: ThinSvdDecomposition<S>,
    svd_work: LinalgWorkspace<'static, S>,
    qr: HouseholderDecomposition<S>,
    qr_work: LinalgWorkspace<'static, S>,
    triangular: Matrix<S>,
    svd_candidate: RandomizedSvdBuffer<S>,
    id_candidate: InterpolativeBuffer<S>,
}
impl<S: NumericScalar> ActionSlot for RandomizedLowRankWorkspace<S> {
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> RandomizedLowRankWorkspace<S> {
    /// Returns the attempt record of the most recent call, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
impl<R: RealScalar> RandomizedCall<R> {
    fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<RandomizedLowRankRequirements<S>, SciOxideError> {
        crate::checked::expect_shape([self.m, self.n], [a.rows(), a.cols()])?;
        let m = self.m;
        let n = self.n;
        let w = self.width;
        let k = self.options.rank;
        let svd =
            ThinSvdDecomposition::<S>::workspace_requirements(&crate::SvdWorkspaceRequest::new(
                w,
                n,
                &super::projected_execution_options(self.options.execution),
            ))?;
        let qr = HouseholderDecomposition::<S>::workspace_requirements(
            &crate::HouseholderWorkspaceRequest::factor(
                w,
                n,
                HouseholderOrientation::Qr,
                &super::projected_execution_options(self.options.execution),
            ),
        )?;
        let payload = sum([
            bytes::<S>(sum([
                w,
                product(&[m, w])?,
                product(&[6, n, w])?,
                product(&[w, w])? - product(&[w, w.saturating_sub(1)])? / 2,
                product(&[2, m])?,
                n,
                product(&[w, w])?,
                w,
                product(&[2, k, sum([m, n])?])?,
                a.apply_workspace_len()?,
                a.apply_adjoint_workspace_len()?,
            ])?)?,
            bytes::<R>(sum([w, k])?)?,
            bytes::<usize>(sum([n, k])?)?,
        ])?;
        RandomizedLowRankRequirements::new(
            RandomizedLayout {
                rows: self.m,
                cols: self.n,
                width: self.width,
                rank: self.options.rank,
                forward: a.apply_workspace_len()?,
                adjoint: a.apply_adjoint_workspace_len()?,
            },
            self.options.execution,
            payload,
            [svd, qr],
            Default::default(),
        )
    }
    fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<RandomizedLowRankWorkspace<S>, SciOxideError> {
        let requirements = self.workspace_requirements(a)?;
        let mut workspace = RandomizedLowRankWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    /// Validate the caller's `n x width` sketch and `n`-row held-out probes, at
    /// least one, with finite entries, before an epoch begins.
    fn sketch_preflight<S: NumericScalar<Real = R>>(
        &self,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
    ) -> Result<(), SciOxideError> {
        crate::checked::expect_shape([self.n, self.width], [sketch.rows(), sketch.cols()])?;
        crate::checked::expect_shape([self.n], [probes.rows()])?;
        if probes.cols() == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("probes"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "at least one held-out probe is required",
            ));
        }
        for view in [sketch, probes] {
            for i in 0..view.rows() {
                for j in 0..view.cols() {
                    crate::checked::validate_input(view.get_element(i, j), "sketch/probe")?;
                }
            }
        }
        Ok(())
    }
    fn range<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<core::ops::ControlFlow<(), usize>, SciOxideError> {
        if w.forward.len() != a.apply_workspace_len()?
            || w.adjoint.len() != a.apply_adjoint_workspace_len()?
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("a"),
                "operator changed its scratch requirement after preparation",
            ));
        }
        // Halko-Martinsson-Tropp range finder: `Q = orth(A Omega)` for the
        // caller's sketch Omega, followed by `power_steps` rounds of
        // subspace iteration `Q = orth(A orth(A^H Q))`. Orthonormalizing
        // after every application (instead of forming `(A A^H)^q A Omega`)
        // prevents the small singular directions from being lost to
        // rounding. Dependent columns are dropped, so `rank` may be below
        // the sketch width. Finally `small = Q^H A` is formed row-wise
        // from `A^H q_j`.
        work.execution.check()?;
        w.q.fill(S::ZERO);
        w.z.fill(S::ZERO);
        let mut rank = 0;
        for column in 0..self.width {
            if !work.iteration()? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            for i in 0..self.n {
                w.right[i] = sketch.get_element(i, column);
            }
            if !work.apply(a, &w.right, &mut w.left, &mut w.forward)? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            append(
                &mut w.left,
                &mut w.q,
                self.m,
                &mut rank,
                &mut w.coefficients,
                self.options.rank_policy,
                work.kernels,
            )?;
        }
        for _ in 0..self.options.power_steps {
            if rank == 0 {
                break;
            }
            if !work.iteration()? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            let mut right_rank = 0;
            for j in 0..rank {
                if !work.adjoint(
                    a,
                    &w.q[j * self.m..(j + 1) * self.m],
                    &mut w.right,
                    &mut w.adjoint,
                )? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
                append(
                    &mut w.right,
                    &mut w.z,
                    self.n,
                    &mut right_rank,
                    &mut w.coefficients,
                    self.options.rank_policy,
                    work.kernels,
                )?;
            }
            rank = 0;
            w.q.fill(S::ZERO);
            for j in 0..right_rank {
                if !work.apply(
                    a,
                    &w.z[j * self.n..(j + 1) * self.n],
                    &mut w.left,
                    &mut w.forward,
                )? {
                    return Ok(core::ops::ControlFlow::Break(()));
                }
                append(
                    &mut w.left,
                    &mut w.q,
                    self.m,
                    &mut rank,
                    &mut w.coefficients,
                    self.options.rank_policy,
                    work.kernels,
                )?;
            }
        }
        w.small.as_mut_slice().fill(S::ZERO);
        for j in 0..rank {
            if !work.adjoint(
                a,
                &w.q[j * self.m..(j + 1) * self.m],
                &mut w.right,
                &mut w.adjoint,
            )? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            for i in 0..self.n {
                w.small.as_mut_slice()[j * self.n + i] = w.right[i].conj();
            }
        }
        Ok(core::ops::ControlFlow::Continue(rank))
    }
    fn svd<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
    ) -> Result<(RandomizedSvdBuffer<S>, LowRankReport<R>), SciOxideError> {
        let mut workspace = self.workspace(a)?;
        let mut output = RandomizedSvdBuffer::new(self.m, self.n, self.options.rank)?;
        let report = self.svd_into(
            a,
            sketch,
            probes,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn svd_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut RandomizedSvdBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        w.slot.reject();
        let requirements = self.workspace_requirements(a)?;
        self.sketch_preflight(sketch, probes)?;
        let k = self.options.rank;
        crate::checked::expect_shape([self.m, k], [out.u.rows(), out.u.cols()])?;
        crate::checked::expect_shape([k, self.n], [out.vt.rows(), out.vt.cols()])?;
        crate::checked::expect_shape([k], [out.singular_values.len()])?;
        w.prepare(&requirements)?;
        run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "RandomizedSVD", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| self.svd_run(a, sketch, probes, out, w, work),
            |report| report.termination() == SpectralTermination::Converged,
        )
    }

    fn svd_run<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut RandomizedSvdBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        let execution = work.execution;
        let k = self.options.rank;
        // `A ~ Q (Q^H A) = (Q U_s) S V^H` from the small dense SVD of `Q^H A`;
        // the kept rank is capped by the requested rank, the range rank and
        // the numerical rank of the small factor.
        let mut report = empty_report();
        let outcome = (|| {
            let core::ops::ControlFlow::Continue(range_rank) =
                self.range(a, sketch, w, &mut work)?
            else {
                report.termination = SpectralTermination::EvaluationLimit;
                return Ok(());
            };
            crate::ThinSvdDecomposition::compute_into(
                &w.small,
                &super::projected_execution_options(self.options.execution),
                &mut w.svd,
                &mut w.svd_work,
                work.kernels.scoped(&mut *work.solve),
            )?;
            let rank = range_rank
                .min(k)
                .min(w.svd.analyze_rank(self.options.rank_policy)?.rank());
            w.svd_candidate.u.as_mut_slice().fill(S::ZERO);
            w.svd_candidate.vt.as_mut_slice().fill(S::ZERO);
            w.svd_candidate.singular_values.fill(R::ZERO);
            for j in 0..rank {
                w.svd_candidate.singular_values[j] = w.svd.singular_values()[j];
                subspace::combine(
                    &mut w.left,
                    &w.q[..range_rank * self.m],
                    (0..range_rank).map(|r| w.svd.u_matrix().get_element(r, j)),
                    core::convert::identity,
                    work.kernels.dispatch(),
                )?;
                for (i, &value) in w.left.iter().enumerate() {
                    w.svd_candidate.u.as_mut_slice()[i * k + j] = value;
                }
                for i in 0..self.n {
                    w.svd_candidate.vt.as_mut_slice()[j * self.n + i] =
                        w.svd.vt_matrix().get_element(j, i);
                }
            }
            report.rank = rank;
            self.probes(a, probes, w, &mut work, &mut report, false)?;
            execution.check()
        })();
        report.termination = super::action_termination(outcome, report.termination)?;
        report.applications = work.count;
        if report.rank == 0 {
            w.svd_candidate.u.as_mut_slice().fill(S::ZERO);
            w.svd_candidate.vt.as_mut_slice().fill(S::ZERO);
            w.svd_candidate.singular_values.fill(R::ZERO);
        }

        out.u
            .as_mut_slice()
            .copy_from_slice(w.svd_candidate.u.as_slice());
        out.vt
            .as_mut_slice()
            .copy_from_slice(w.svd_candidate.vt.as_slice());
        out.singular_values
            .copy_from_slice(&w.svd_candidate.singular_values);
        out.rank = report.rank;
        Ok(report)
    }
    fn interpolative<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
    ) -> Result<(InterpolativeBuffer<S>, LowRankReport<R>), SciOxideError> {
        let mut workspace = self.workspace(a)?;
        let mut output = InterpolativeBuffer::new(self.m, self.n, self.options.rank)?;
        let report = self.interpolative_into(
            a,
            sketch,
            probes,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn interpolative_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut InterpolativeBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        w.slot.reject();
        let requirements = self.workspace_requirements(a)?;
        self.sketch_preflight(sketch, probes)?;
        let k = self.options.rank;
        crate::checked::expect_shape([self.m, k], [out.columns.rows(), out.columns.cols()])?;
        crate::checked::expect_shape(
            [k, self.n],
            [out.coefficients.rows(), out.coefficients.cols()],
        )?;
        crate::checked::expect_shape([k], [out.column_indices.len()])?;
        w.prepare(&requirements)?;
        run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "Interpolative", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| self.interpolative_run(a, sketch, probes, out, w, work),
            |report| report.termination() == SpectralTermination::Converged,
        )
    }

    fn interpolative_run<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut InterpolativeBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        let execution = work.execution;
        let k = self.options.rank;
        // Column interpolative decomposition: a column-pivoted QR of the
        // small `Q^H A = [R11 R12] P^T` selects k skeleton columns of A, and
        // the interpolation coefficients `[I, R11^{-1} R12]` (scattered back
        // through P) express every column in terms of them. The skeleton
        // columns are then fetched exactly as `A e_j`.
        let mut report = empty_report();
        let outcome = (|| {
            let core::ops::ControlFlow::Continue(range_rank) =
                self.range(a, sketch, w, &mut work)?
            else {
                report.termination = SpectralTermination::EvaluationLimit;
                return Ok(());
            };
            HouseholderDecomposition::factor_into(
                &w.small,
                HouseholderOrientation::Qr,
                crate::HouseholderPivoting::Column,
                &super::projected_execution_options(self.options.execution),
                &mut w.qr,
                &mut w.qr_work,
                work.kernels.scoped(&mut *work.solve),
            )?;
            w.qr.qr_triangular_into(&mut w.triangular)?;
            let rank = range_rank
                .min(k)
                .min(w.qr.analyze_rank(self.options.rank_policy)?.rank());
            w.id_candidate.columns.as_mut_slice().fill(S::ZERO);
            w.id_candidate.coefficients.as_mut_slice().fill(S::ZERO);
            for j in 0..self.n {
                let original = w.qr.permutation()[j];
                if j < rank {
                    w.id_candidate.coefficients.as_mut_slice()[j * self.n + original] = S::ONE;
                    continue;
                }
                for i in (0..rank).rev() {
                    let mut value = w.triangular.get_element(i, j);
                    for u in i + 1..rank {
                        value = value
                            - w.triangular.get_element(i, u)
                                * w.id_candidate.coefficients.get_element(u, original);
                    }
                    let diagonal = w.triangular.get_element(i, i);
                    if diagonal == S::ZERO {
                        return Err(scioxide_core::SciOxideError::singular(None));
                    }
                    w.id_candidate.coefficients.as_mut_slice()[i * self.n + original] =
                        crate::checked::finite_candidate(value / diagonal)?;
                }
            }
            for j in 0..rank {
                let index = w.qr.permutation()[j];
                w.id_candidate.column_indices[j] = index;
                w.right.fill(S::ZERO);
                w.right[index] = S::ONE;
                if !work.apply(a, &w.right, &mut w.left, &mut w.forward)? {
                    report.termination = SpectralTermination::EvaluationLimit;
                    return Ok(());
                }
                for i in 0..self.m {
                    w.id_candidate.columns.as_mut_slice()[i * k + j] = w.left[i];
                }
            }
            report.rank = rank;
            self.probes(a, probes, w, &mut work, &mut report, true)?;
            execution.check()
        })();
        report.termination = super::action_termination(outcome, report.termination)?;
        report.applications = work.count;
        if report.rank == 0 {
            w.id_candidate.columns.as_mut_slice().fill(S::ZERO);
            w.id_candidate.coefficients.as_mut_slice().fill(S::ZERO);
            w.id_candidate.column_indices.fill(0);
        }

        out.columns
            .as_mut_slice()
            .copy_from_slice(w.id_candidate.columns.as_slice());
        out.coefficients
            .as_mut_slice()
            .copy_from_slice(w.id_candidate.coefficients.as_slice());
        out.column_indices
            .copy_from_slice(&w.id_candidate.column_indices);
        out.rank = report.rank;
        Ok(report)
    }

    fn probes<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        probes: MatrixView<'_, S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        report: &mut LowRankReport<R>,
        id: bool,
    ) -> Result<(), SciOxideError> {
        let rank = report.rank;
        if self.m == 0 || self.n == 0 {
            report.termination = SpectralTermination::Converged;
            return Ok(());
        }
        // A posteriori accuracy check with held-out probe vectors that were
        // not used to build the range: `||A g - A_k g||` compared against
        // `||A g||` estimates the approximation error independently of the
        // construction, which is what makes the reported termination honest.
        let mut residual = R::ZERO;
        let mut relative = None;
        let mut accepted = true;
        for probe in 0..probes.cols() {
            for i in 0..self.n {
                w.right[i] = probes.get_element(i, probe);
            }
            if !work.apply(a, &w.right, &mut w.left, &mut w.forward)? {
                report.termination = SpectralTermination::EvaluationLimit;
                return Ok(());
            }
            let reference = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.left),
                work.kernels,
            )?;
            w.approximation.fill(S::ZERO);
            for j in 0..rank {
                let mut coefficient = S::ZERO;
                for i in 0..self.n {
                    coefficient = coefficient
                        + (if id {
                            w.id_candidate.coefficients.get_element(j, i)
                        } else {
                            w.svd_candidate.vt.get_element(j, i)
                        }) * w.right[i];
                }
                if !id {
                    coefficient = coefficient * S::from_real(w.svd_candidate.singular_values[j]);
                }
                for i in 0..self.m {
                    w.approximation[i] = w.approximation[i]
                        + coefficient
                            * if id {
                                w.id_candidate.columns.get_element(i, j)
                            } else {
                                w.svd_candidate.u.get_element(i, j)
                            };
                }
            }
            for i in 0..self.m {
                w.approximation[i] = crate::checked::sub(w.left[i], w.approximation[i])?;
            }
            let error = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.approximation),
                work.kernels,
            )?;
            residual = residual.max(error);
            accepted &= error <= self.options.tolerance.threshold(reference)?;
            if reference > R::ZERO {
                let ratio = error / reference;
                relative = Some(relative.map_or(ratio, |r: R| r.max(ratio)));
            }
            report.probe_residual = residual;
            report.relative_probe_residual = relative;
            report.completed_probes += 1;
        }
        report.termination = if accepted {
            SpectralTermination::Converged
        } else {
            SpectralTermination::AccuracyLimit
        };
        Ok(())
    }
}
impl<R: RealScalar> RandomizedLowRank<R> {
    /// Checks the settings against `m × n` operators and sketches of `width`
    /// columns and returns the per-call value. `sketch` names the argument that
    /// supplies the width.
    fn call(
        &self,
        m: usize,
        n: usize,
        width: usize,
        sketch: scioxide_core::Argument,
    ) -> Result<RandomizedCall<R>, SciOxideError> {
        if self.rank > width || width > m.min(n) {
            let argument = if self.rank > width {
                scioxide_core::argument!("self.rank")
            } else {
                sketch
            };
            return Err(crate::checked::out_of_range(
                argument,
                "low-rank target/sketch width exceeds dimensions",
            ));
        }
        Ok(RandomizedCall {
            m,
            n,
            width,
            options: *self,
        })
    }
    /// Returns the workspace storage needed to compute a low-rank approximation
    /// of `a` from sketches of `width` columns.
    ///
    /// The operator is not applied; only its shape and scratch lengths are read.
    /// The storage includes the dense SVD and QR. The settings are checked
    /// first, so the execution methods report the same errors before any work.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the target rank exceeds `width`
    ///   (argument `self.rank`) or `width` exceeds `min(m, n)` (argument
    ///   `width`, or `sketch` in the execution methods).
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of `a`.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        width: usize,
    ) -> Result<RandomizedLowRankRequirements<S>, SciOxideError> {
        self.call(a.rows(), a.cols(), width, scioxide_core::argument!("width"))?
            .workspace_requirements(a)
    }

    /// Allocates a workspace for `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        width: usize,
    ) -> Result<RandomizedLowRankWorkspace<S>, SciOxideError> {
        self.call(a.rows(), a.cols(), width, scioxide_core::argument!("width"))?
            .workspace(a)
    }

    /// Computes the approximate SVD with a new workspace and result buffer.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`RandomizedSvdBuffer::new`] and
    /// [`Self::svd_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, RandomizedLowRank};
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0., 0., 0.])?;
    /// let sketch = Matrix::from_vec(2, 1, vec![1., 1.])?;
    /// let probes = Matrix::from_vec(2, 1, vec![1., 0.])?;
    /// let method = RandomizedLowRank::new(1);
    /// let (output, report) = method.svd(&a.view().try_finite()?, sketch.view(), probes.view())?;
    /// // A = diag(2, 0) has the single nonzero singular value 2.
    /// assert!((output.singular_values()[0] - 2.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn svd<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
    ) -> Result<(RandomizedSvdBuffer<S>, LowRankReport<R>), SciOxideError> {
        self.call(
            a.rows(),
            a.cols(),
            sketch.cols(),
            scioxide_core::argument!("sketch"),
        )?
        .svd(a, sketch, probes)
    }

    /// Computes the approximate SVD `A ≈ U Σ Vᴴ` and writes it to `out`.
    ///
    /// `sketch` is the `n × width` test matrix and `probes` holds one probe
    /// vector of length `n` per column.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `sketch` is not `n × width`, `probes` does not
    ///   have `n` rows, or `out` was allocated for another shape or rank.
    /// - `InvalidInput` (`OutOfRange`) if `probes` has no columns.
    /// - `NonFinite` if `sketch` or `probes` has a non-finite entry, or if `a`
    ///   returns a non-finite value.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `InvalidInput` (`Inconsistent`) if the scratch lengths of `a` changed
    ///   after the workspace was reserved.
    /// - `NumericalInstability` (`Overflow`) if a probe residual overflows.
    /// - Errors of the dense SVD, and errors returned by `a`.
    pub fn svd_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut RandomizedSvdBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        self.call(
            a.rows(),
            a.cols(),
            sketch.cols(),
            scioxide_core::argument!("sketch"),
        )?
        .svd_into(a, sketch, probes, out, w, context)
    }

    /// Computes the interpolative decomposition with a new workspace and result buffer.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`InterpolativeBuffer::new`] and
    /// [`Self::interpolative_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, RandomizedLowRank};
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0., 0., 0.])?;
    /// let sketch = Matrix::from_vec(2, 1, vec![1., 1.])?;
    /// let probes = Matrix::from_vec(2, 1, vec![1., 0.])?;
    /// let method = RandomizedLowRank::new(1);
    /// let (output, report) =
    ///     method.interpolative(&a.view().try_finite()?, sketch.view(), probes.view())?;
    /// // Only column 0 of A = diag(2, 0) is nonzero.
    /// assert_eq!(output.column_indices(), [0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn interpolative<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
    ) -> Result<(InterpolativeBuffer<S>, LowRankReport<R>), SciOxideError> {
        self.call(
            a.rows(),
            a.cols(),
            sketch.cols(),
            scioxide_core::argument!("sketch"),
        )?
        .interpolative(a, sketch, probes)
    }

    /// Computes a column interpolative decomposition `A ≈ C Z` and writes it to `out`.
    ///
    /// A column-pivoted QR factorization `Qᴴ A P = Q̃ [R₁₁ R₁₂]` selects the
    /// columns, and `Z = [I, R₁₁⁻¹ R₁₂] Pᵀ`. The selected columns `C` are then
    /// computed as `A eⱼ` with one product each. `sketch` and `probes` are as
    /// in [`Self::svd_into`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::svd_into`], with the dense pivoted QR in place of
    ///   the dense SVD.
    /// - `Singular` if a diagonal entry of `R₁₁` is exactly zero.
    pub fn interpolative_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        sketch: MatrixView<'_, S>,
        probes: MatrixView<'_, S>,
        out: &mut InterpolativeBuffer<S>,
        w: &mut RandomizedLowRankWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<LowRankReport<R>, SciOxideError> {
        self.call(
            a.rows(),
            a.cols(),
            sketch.cols(),
            scioxide_core::argument!("sketch"),
        )?
        .interpolative_into(a, sketch, probes, out, w, context)
    }
}

fn append<S: NumericScalar>(
    vector: &mut [S],
    basis: &mut [S],
    n: usize,
    rank: &mut usize,
    coefficients: &mut [S],
    policy: RankPolicy<S::Real>,
    kernels: crate::KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let original = crate::reduction::l2_norm_with_context(VectorView::from_slice(vector), kernels)?;
    let norm = subspace::project(
        vector,
        &basis[..n * *rank],
        coefficients,
        |_, _| {},
        kernels,
    )?;
    if norm > policy.cutoff(original, n, *rank + 1)? {
        for (i, &x) in vector.iter().enumerate() {
            basis[*rank * n + i] = x / S::from_real(norm);
        }
        *rank += 1;
    }
    Ok(())
}

impl<S: NumericScalar> RandomizedSvdBuffer<S> {
    fn empty() -> Self {
        Self {
            u: Matrix::empty(),
            singular_values: Vec::new(),
            vt: Matrix::empty(),
            rank: 0,
        }
    }
}

impl<S: NumericScalar> StorageBytes for RandomizedSvdBuffer<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.u.storage_bytes()?,
            self.singular_values.storage_bytes()?,
            self.vt.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> InterpolativeBuffer<S> {
    fn empty() -> Self {
        Self {
            columns: Matrix::empty(),
            coefficients: Matrix::empty(),
            column_indices: Vec::new(),
            rank: 0,
        }
    }
}

impl<S: NumericScalar> StorageBytes for InterpolativeBuffer<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.columns.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.column_indices.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> Default for RandomizedLowRankWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> RandomizedLowRankWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            slot: SolveSlot::new(),
            m: 0,
            n: 0,
            width: 0,
            q: Vec::new(),
            z: Vec::new(),
            left: Vec::new(),
            coefficients: Vec::new(),
            right: Vec::new(),
            approximation: Vec::new(),
            forward: Vec::new(),
            adjoint: Vec::new(),
            small: Matrix::empty(),
            svd: ThinSvdDecomposition::empty(),
            svd_work: LinalgWorkspace::new(),
            qr: HouseholderDecomposition::empty(),
            qr_work: LinalgWorkspace::new(),
            triangular: Matrix::empty(),
            svd_candidate: RandomizedSvdBuffer::empty(),
            id_candidate: InterpolativeBuffer::empty(),
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
        requirements: &RandomizedLowRankRequirements<S>,
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
        requirements: &RandomizedLowRankRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for RandomizedLowRankWorkspace<S> {
    type Requirements = RandomizedLowRankRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &RandomizedLowRankRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let RandomizedLayout {
            rows: m,
            cols: n,
            width,
            rank,
            forward,
            adjoint,
        } = requirements.layout;

        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.q, product(&[m, width])?)?;
        plan.buffer(&self.z, product(&[n, width])?)?;
        plan.buffer(&self.coefficients, width)?;
        plan.buffer(&self.left, m)?;
        plan.buffer(&self.right, n)?;
        plan.buffer(&self.approximation, m)?;
        plan.buffer(&self.forward, forward)?;
        plan.buffer(&self.adjoint, adjoint)?;
        plan.matrix(&self.small, width, n)?;
        plan.matrix(&self.triangular, width, n)?;
        self.svd.action_storage(&mut plan, width, n)?;
        self.qr.action_storage(&mut plan, width, n)?;
        plan.dense(&self.svd_work, &requirements.storage.dense[0])?;
        plan.dense(&self.qr_work, &requirements.storage.dense[1])?;
        plan.matrix(&self.svd_candidate.u, m, rank)?;
        plan.matrix(&self.svd_candidate.vt, rank, n)?;
        plan.buffer(&self.svd_candidate.singular_values, rank)?;
        plan.matrix(&self.id_candidate.columns, m, rank)?;
        plan.matrix(&self.id_candidate.coefficients, rank, n)?;
        plan.buffer(&self.id_candidate.column_indices, rank)?;
        Ok(plan)
    }
    fn configure(
        &mut self,
        requirements: &RandomizedLowRankRequirements<S>,
    ) -> Result<(), SciOxideError> {
        let RandomizedLayout {
            rows: m,
            cols: n,
            width,
            rank,
            forward,
            adjoint,
        } = requirements.layout;

        resize(&mut self.q, product(&[m, width])?, S::ZERO)?;
        resize(&mut self.z, product(&[n, width])?, S::ZERO)?;
        resize(&mut self.coefficients, width, S::ZERO)?;
        resize(&mut self.left, m, S::ZERO)?;
        resize(&mut self.right, n, S::ZERO)?;
        resize(&mut self.approximation, m, S::ZERO)?;
        resize(&mut self.forward, forward, S::ZERO)?;
        resize(&mut self.adjoint, adjoint, S::ZERO)?;
        self.small.try_resize_flat(width, n)?;
        self.triangular.try_resize_flat(width, n)?;
        self.svd.reserve_result(width, n)?;
        self.qr
            .try_reserve(width, n, crate::HouseholderOrientation::Qr)?;
        self.svd_work.try_reserve(&requirements.storage.dense[0])?;
        self.qr_work.try_reserve(&requirements.storage.dense[1])?;
        self.svd_candidate.u.try_resize_flat(m, rank)?;
        self.svd_candidate.vt.try_resize_flat(rank, n)?;
        resize(&mut self.svd_candidate.singular_values, rank, S::Real::ZERO)?;
        self.id_candidate.columns.try_resize_flat(m, rank)?;
        self.id_candidate.coefficients.try_resize_flat(rank, n)?;
        resize(&mut self.id_candidate.column_indices, rank, 0)?;
        self.m = m;
        self.n = n;
        self.width = width;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for RandomizedLowRankWorkspace<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.q.storage_bytes()?,
            self.z.storage_bytes()?,
            self.left.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.right.storage_bytes()?,
            self.approximation.storage_bytes()?,
            self.forward.storage_bytes()?,
            self.adjoint.storage_bytes()?,
            self.small.storage_bytes()?,
            self.svd.storage_bytes()?,
            self.svd_work.storage_bytes()?,
            self.qr.storage_bytes()?,
            self.qr_work.storage_bytes()?,
            self.triangular.storage_bytes()?,
            self.svd_candidate.storage_bytes()?,
            self.id_candidate.storage_bytes()?,
        ])
    }
}

impl<R: RealScalar> LowRankReport<R> {
    /// Returns the termination status.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }

    /// Returns the rank `r` of the approximation.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the largest probe residual `‖A g − Ã g‖₂` over the completed probes.
    pub const fn probe_residual(&self) -> R {
        self.probe_residual
    }

    /// Returns the largest `‖A g − Ã g‖₂ / ‖A g‖₂` over the completed probes with
    /// `A g ≠ 0`, or `None` if there are none.
    pub const fn relative_probe_residual(&self) -> Option<R> {
        self.relative_probe_residual
    }

    /// Returns the number of probes whose residual was computed.
    pub const fn completed_probes(&self) -> usize {
        self.completed_probes
    }

    /// Returns the number of forward plus adjoint operator applications.
    pub const fn applications(&self) -> usize {
        self.applications
    }
}

impl<S: NumericScalar> RandomizedSvdBuffer<S> {
    /// Returns the left singular vectors as the columns of an `m × r` view.
    pub fn u(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.u.view();
        view.cols = self.rank;
        view
    }
    /// Returns `Vᴴ` as an `r × n` view; its rows are the conjugated right singular vectors.
    pub fn vt(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.vt.view();
        view.rows = self.rank;
        view
    }
    /// Returns the `r` singular values in descending order.
    pub fn singular_values(&self) -> &[S::Real] {
        &self.singular_values[..self.rank]
    }
}

impl<S: NumericScalar> InterpolativeBuffer<S> {
    /// Returns the selected columns `C` as an `m × r` view.
    pub fn columns(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.columns.view();
        view.cols = self.rank;
        view
    }
    /// Returns the interpolation matrix `Z` as an `r × n` view, in the column order of `A`.
    pub fn coefficients(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.coefficients.view();
        view.rows = self.rank;
        view
    }
    /// Returns the indices of the selected columns of `A`.
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices[..self.rank]
    }
}
