//! Minimum-norm least-squares solutions from a complete orthogonal decomposition
//! built on pivoted QR (or the QR of the adjoint, for LQ).
use crate::workspace::ScalarRegion;
use crate::{
    ConditionEstimate, DirectSolveOptions, DirectSolveReport, DirectSolveResult,
    DirectSolveTermination, Estimation, HouseholderDecomposition, HouseholderOrientation,
    KernelContext, LinalgRequirements, LinalgWorkspace, Matrix, MatrixView, MatrixViewMut,
    RankReport, Vector, VectorView, VectorViewMut,
};
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, ScaledValue,
    SciOxideError,
};

/// Plan for minimum-norm least-squares solutions `x = A_r⁺ b` of a possibly
/// rank-deficient or underdetermined system, created by
/// [`HouseholderDecomposition::into_minimum_norm`].
///
/// The numerical rank `r` is the number of leading diagonal entries of the pivoted `R` whose
/// modulus exceeds the threshold of the rank policy (for LQ, `R` is the triangular factor of `Aᴴ`,
/// and the construction below applies to `Aᴴ`). Construction reduces the leading `r` rows of `R` to
/// `[T 0] Z`, with `T` upper triangular and `Z` unitary, once, in O(r² n) operations. This gives
/// the complete orthogonal decomposition `A P = Q [T 0; 0 0] Z` of the rank-`r` truncation `A_r`.
/// Each solve then returns `x = P Zᴴ [T⁻¹ (Qᴴ b)₁; 0]`, the solution of minimum 2-norm among the
/// minimizers of `‖A_r x − b‖₂`, in O(mn + rn + r²) time per right-hand side.
///
/// A rank-deficient plan is not rejected. Its condition estimate is 0, so a
/// positive condition threshold in the options rejects it as `IllConditioned`. The
/// options given to [`HouseholderDecomposition::into_minimum_norm`] also set
/// the residual diagnostic and the limits of every solve.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §5.4 and §5.5.
#[derive(Debug, Clone)]
#[must_use]
pub struct HouseholderMinimumNormPlan<S: NumericScalar> {
    factor: HouseholderDecomposition<S>,
    rz: Matrix<S>,
    tau: Vec<S>,
    maximum: S::Real,
    rank: RankReport<S::Real>,
    acceptance: DirectSolveOptions<S::Real>,
}

impl<S: NumericScalar> HouseholderMinimumNormPlan<S> {
    pub(super) fn prepare(
        factor: HouseholderDecomposition<S>,
        acceptance: DirectSolveOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let options = acceptance
            .execution_options()
            .with_execution_policy(ExecutionPolicy::Sequential);
        // Preparation is an owned, untimed reduction; the retained options bound
        // the later scoped solves.
        let report = factor.analyze_rank(acceptance.rank_policy())?;
        let r = factor
            .triangular_diagonal()
            .iter()
            .take_while(|x| x.abs() > report.threshold())
            .count();
        let rank =
            RankReport::from_factor(r, report.maximum_rank(), report.threshold(), report.scale());
        let packed = factor.qr.packed_factors().as_view();
        let n = packed.cols();
        let width = if r == 0 || r == n { 0 } else { n - r + 1 };
        let mut workspace = LinalgWorkspace::new();
        let requirements = LinalgRequirements::new(options.byte_limit()).with_scalar(
            ScalarRegion::Direct,
            scioxide_core::backend::checked_product(
                &[2, width],
                scioxide_core::quantity!("rz_scratch"),
            )?,
        );
        workspace.try_reserve(&requirements)?;
        let mut rz = Matrix::zeros(if r == n { 0 } else { r }, n)?;
        let mut tau = Vec::new();
        super::util::resize_vec(
            &mut tau,
            if r == n { 0 } else { r },
            S::ZERO,
            scioxide_core::quantity!("rz_reflectors"),
        )?;
        for i in 0..rz.rows() {
            for j in i..n {
                rz.as_mut_slice()[i * n + j] = packed.get_element(i, j);
            }
        }
        let scratch = workspace.scalar_mut(ScalarRegion::Direct);
        if width != 0 {
            let (v, x) = crate::workspace::split_mut(scratch, width)?;
            for i in (0..r).rev() {
                v[0] = rz.as_slice()[i * n + i].conj();
                for (j, value) in v.iter_mut().enumerate().skip(1) {
                    *value = rz.as_slice()[i * n + r + j - 1].conj();
                }
                let reflector = crate::reflector::generate(VectorViewMut::from_slice(v))?;
                tau[i] = reflector.tau;
                let vector = crate::FiniteVectorView::from_workspace(VectorView::from_slice(v));
                for row in 0..i {
                    x[0] = rz.as_slice()[row * n + i].conj();
                    for (j, value) in x.iter_mut().enumerate().take(width).skip(1) {
                        *value = rz.as_slice()[row * n + r + j - 1].conj();
                    }
                    crate::reflector::apply(
                        vector,
                        reflector.tau.conj(),
                        MatrixViewMut::from_slice(width, 1, &mut x[..width])?,
                    )?;
                    rz.as_mut_slice()[row * n + i] = x[0].conj();
                    for (j, &value) in x.iter().enumerate().take(width).skip(1) {
                        rz.as_mut_slice()[row * n + r + j - 1] = value.conj();
                    }
                }
                rz.as_mut_slice()[i * n + i] = S::from_real(reflector.beta);
                rz.as_mut_slice()[i * n + r..(i + 1) * n].copy_from_slice(&v[1..]);
            }
        }
        let maximum = if r == n {
            factor.qr.packed_factors().maximum
        } else {
            rz.as_slice().iter().fold(S::Real::ZERO, |m, &x| {
                m.max(crate::kernel::standard::magnitude(x))
            })
        };
        Ok(Self {
            factor,
            rz,
            tau,
            maximum,
            rank,
            acceptance,
        })
    }

    /// Returns the underlying QR or LQ factorization of `A`, before truncation.
    pub const fn factor(&self) -> &HouseholderDecomposition<S> {
        &self.factor
    }
    /// Returns the numerical rank `r` used by every solve.
    pub const fn rank(&self) -> RankReport<S::Real> {
        self.rank
    }

    /// Returns the bytes of factor storage currently allocated, including the QR
    /// factors and the reflectors of `Z`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let extra = super::util::storage_bytes([
            (self.rz.capacity(), core::mem::size_of::<S>()),
            (self.tau.capacity(), core::mem::size_of::<S>()),
        ])?;
        self.factor
            .retained_bytes()?
            .checked_add(extra)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("minimum_norm_factor"))
            })
    }

    /// Returns the workspace size for solves with `right_hand_sides` columns,
    /// including the condition estimate.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the plan's options.
    pub fn workspace_requirements(
        &self,
        right_hand_sides: usize,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let options = self.acceptance.execution_options();
        let options = &options;
        let (m, n) = (self.factor.rows(), self.factor.cols());
        if right_hand_sides == 0 {
            let mut requirements = if self.acceptance.minimum_reciprocal_condition().is_some() {
                crate::estimate::condition_requirements(m.max(n), options.byte_limit())?
            } else {
                LinalgRequirements::default()
            };
            requirements.byte_limit = options.byte_limit();
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        let count = scioxide_core::backend::checked_product(
            &[n, right_hand_sides],
            scioxide_core::quantity!("minimum_norm_scratch"),
        )?;
        let vectors = scioxide_core::backend::checked_product(
            &[2, m.max(n)],
            scioxide_core::quantity!("minimum_norm_scratch"),
        )?;
        let mut requirements = LinalgRequirements::default();
        *requirements.scalar_mut(ScalarRegion::Direct) =
            count.checked_add(vectors).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("minimum_norm_scratch"))
            })?;
        requirements.scaled = vectors;
        requirements = requirements.max(&crate::estimate::condition_requirements(
            m.max(n),
            options.byte_limit(),
        )?);
        requirements.byte_limit = options.byte_limit();
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }

    /// Allocates a workspace for solves with `right_hand_sides` columns.
    ///
    /// # Errors
    ///
    /// Workspace errors as described in
    /// [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn workspace(
        &self,
        right_hand_sides: usize,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(right_hand_sides)?)?;
        Ok(workspace)
    }

    /// Returns the minimum-norm least-squares solution of `A_r x = b`.
    ///
    /// The solve runs on the calling thread. It is rejected only by a
    /// condition threshold; see [`HouseholderMinimumNormPlan`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have length `m`.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `DomainError` if `A` has no rows or columns and a condition threshold is set.
    /// - `NumericalInstability` (`Overflow`) or `NonFinite` if an entry of the
    ///   solution overflows.
    /// - Limit errors from the plan's options as described in
    ///   [Execution](crate#execution).
    pub fn solve<'b>(
        &self,
        rhs: impl Into<VectorView<'b, S>>,
    ) -> Result<DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let mut output = Vector::zeros(self.factor.cols())?;
        let report =
            self.solve_matrix_impl(
                crate::MatrixView::from(rhs.into()),
                crate::MatrixViewMut::from(output.view_mut()),
                &mut LinalgWorkspace::new(),
                crate::LinalgExecutionContext::new(
                    scioxide_core::ExecutionContext::sequential_native(),
                ),
                &self
                    .acceptance
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            )?;
        Ok(DirectSolveResult::new(output, report))
    }
    /// Writes the minimum-norm solution to `output`, the reusable form of [`Self::solve`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have length `n`.
    /// - The errors of [`Self::solve`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<VectorView<'b, S>>,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        let rhs = rhs.into();
        let mut matrix =
            MatrixView::from_slice_with_strides(rhs.len(), 1, rhs.data, rhs.stride(), 1)?;
        if rhs.is_conjugated() {
            matrix = matrix.conjugate();
        }
        let output =
            MatrixViewMut::from_slice_with_strides(output.len(), 1, output.data, output.stride, 1)?;
        self.solve_matrix_into(matrix, output, workspace, context)
    }
    /// Returns the minimum-norm solution of `A_r X = B` for every column of `rhs`.
    ///
    /// A requested residual is the largest backward error over the columns.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `m` rows.
    /// - The other errors of [`Self::solve`].
    pub fn solve_matrix<'b>(
        &self,
        rhs: impl Into<MatrixView<'b, S>>,
    ) -> Result<DirectSolveResult<Matrix<S>, S::Real>, SciOxideError> {
        let rhs = rhs.into();
        let mut output = Matrix::zeros(self.factor.cols(), rhs.cols())?;
        let report =
            self.solve_matrix_impl(
                rhs,
                output.view_mut(),
                &mut LinalgWorkspace::new(),
                crate::LinalgExecutionContext::new(
                    scioxide_core::ExecutionContext::sequential_native(),
                ),
                &self
                    .acceptance
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            )?;
        Ok(DirectSolveResult::new(output, report))
    }
    /// Writes the minimum-norm solutions to `output`, the reusable form of
    /// [`Self::solve_matrix`].
    ///
    /// Each column counts toward the iteration budget of the plan's options.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `n × rhs.cols()`.
    /// - The errors of [`Self::solve_matrix`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn solve_matrix_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<MatrixView<'b, S>>,
        output: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        self.solve_matrix_impl(
            rhs.into(),
            output,
            workspace,
            context.into(),
            &self.acceptance.execution_options(),
        )
    }

    fn solve_matrix_impl(
        &self,
        rhs: MatrixView<'_, S>,
        mut output: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: crate::LinalgExecutionContext<'_, '_>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.with_options(options)?;
        context.execution().check()?;
        let (m, n) = (self.factor.rows(), self.factor.cols());
        if rhs.rows() != m {
            return Err(SciOxideError::dimension_mismatch(
                [m, rhs.cols()],
                [rhs.rows(), rhs.cols()],
            ));
        }
        if output.rows() != n || output.cols() != rhs.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [n, rhs.cols()],
                [output.rows(), output.cols()],
            ));
        }
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        workspace.try_reserve(&self.workspace_requirements(rhs.cols())?)?;
        if self.rank.maximum_rank() == 0 && self.acceptance.minimum_reciprocal_condition().is_some()
        {
            return Err(crate::analysis::empty_condition());
        }
        let count = n * rhs.cols();
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "householder_minimum_norm", 1),
            ((m as u128) << 64) | n as u128,
            0,
        );
        workspace.run_scope(
            context,
            identity,
            options.solve_budgets(),
            |workspace, kernels, solve| {
                let condition = self
                    .acceptance
                    .minimum_reciprocal_condition()
                    .map(|_| self.condition_estimate_scoped(workspace, kernels.scoped(&mut *solve)))
                    .transpose()?;
                if self
                    .acceptance
                    .minimum_reciprocal_condition()
                    .is_some_and(|threshold| {
                        condition
                            .is_some_and(|estimate| estimate.reciprocal().value() < threshold.get())
                    })
                {
                    return Ok(DirectSolveReport::new(
                        DirectSolveTermination::IllConditioned,
                        self.rank,
                        condition,
                        None,
                    ));
                }
                if rhs.cols() == 0 || (m == 0 && n == 0) {
                    return Ok(DirectSolveReport::new(
                        DirectSolveTermination::Accepted,
                        self.rank,
                        condition,
                        crate::analysis::empty_residual(&self.acceptance, rhs.cols())?,
                    ));
                }

                workspace.with_prefix(count + 2 * m.max(n), |storage, _| {
                    let (solution, rest) = crate::workspace::split_mut(storage, count)?;
                    let (work, scratch) = crate::workspace::split_mut(rest, m.max(n))?;
                    for col in 0..rhs.cols() {
                        crate::solve_support::admit_iteration(
                            Some(kernels.execution()),
                            Some(&mut *solve),
                            "minimum-norm RHS budget exhausted",
                        )?;
                        self.solve_column(rhs.as_view().col(col)?, work, scratch, kernels)?;
                        for row in 0..n {
                            solution[row * rhs.cols() + col] = work[row];
                        }
                    }
                    let residual = if self.acceptance.residual() == Estimation::Estimate {
                        self.factor.residual_columns(
                            rhs.as_view(),
                            MatrixView::from_slice(n, rhs.cols(), solution)?,
                            kernels,
                        )?
                    } else {
                        None
                    };
                    crate::checked::validate_values(
                        solution,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                        "minimum-norm solution is not finite",
                    )?;
                    kernels.execution().check()?;
                    for row in 0..n {
                        for col in 0..rhs.cols() {
                            output.set_element(row, col, solution[row * rhs.cols() + col]);
                        }
                    }
                    Ok(DirectSolveReport::new(
                        DirectSolveTermination::Accepted,
                        self.rank,
                        condition,
                        residual,
                    ))
                })
            },
        )
    }

    fn triangular(&self) -> MatrixView<'_, S> {
        if self.rz.rows() == 0 {
            self.factor.qr.packed_factors().as_view()
        } else {
            self.rz.view()
        }
    }

    fn solve_column(
        &self,
        rhs: VectorView<'_, S>,
        work: &mut [S],
        scratch: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        work.fill(S::ZERO);
        let r = self.rank.rank();
        if r == 0 {
            return Ok(());
        }
        let normal = self.factor.orientation() == HouseholderOrientation::Qr;
        if normal {
            for (x, b) in work.iter_mut().zip(rhs.iter()) {
                *x = b;
            }
            self.factor.qr.apply_qh_vector_in_place(work)?;
        } else {
            for (i, &original) in self.factor.permutation().iter().enumerate() {
                work[i] = rhs.get_element(original);
            }
            self.apply_z(work, scratch, true)?;
        }
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: crate::kernel::triangular::Storage::Dense(
                    self.triangular().block(0, 0, r, r)?,
                ),
                triangle: scioxide_core::Triangle::Upper,
                op: if normal {
                    crate::kernel::GemmOp::None
                } else {
                    crate::kernel::GemmOp::Adjoint
                },
                diagonal: scioxide_core::DiagonalKind::Explicit,
            },
            self.maximum,
            &mut work[..r],
            r.saturating_sub(1),
            context.dispatch(),
        )?;
        work[r..].fill(S::ZERO);
        if normal {
            self.apply_z(work, scratch, false)?;
            scratch[..self.factor.cols()].copy_from_slice(&work[..self.factor.cols()]);
            for (i, &original) in self.factor.permutation().iter().enumerate() {
                work[original] = scratch[i];
            }
        } else {
            self.factor.qr.apply_q_vector_in_place(work)?;
        }
        Ok(())
    }

    // R W = [T 0], with W = H_(r-1) ... H_0. Applying W runs H_0 first.
    fn apply_z(
        &self,
        values: &mut [S],
        scratch: &mut [S],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        let (r, n) = (self.rank.rank(), self.rz.cols());
        if r == 0 || r == n {
            return Ok(());
        }
        let width = n - r + 1;
        let scratch = &mut scratch[..width];
        for step in 0..r {
            let i = if adjoint { r - 1 - step } else { step };
            scratch[0] = values[i];
            scratch[1..].copy_from_slice(&values[r..n]);
            let vector = VectorView::from_slice(&self.rz.as_slice()[i * n + r - 1..(i + 1) * n]);
            crate::reflector::apply(
                crate::FiniteVectorView::from_workspace(vector),
                if adjoint {
                    self.tau[i].conj()
                } else {
                    self.tau[i]
                },
                MatrixViewMut::from_slice(width, 1, scratch)?,
            )?;
            values[i] = scratch[0];
            values[r..n].copy_from_slice(&scratch[1..]);
        }
        Ok(())
    }

    /// Estimates the reciprocal condition number `1 / (‖A‖₁ ‖A_r⁺‖₁)`.
    ///
    /// The method is that of
    /// [`LuDecomposition::condition_estimate`](crate::LuDecomposition::condition_estimate).
    /// The estimate is 0 when `r < min(m, n)`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `A` has no rows or columns.
    /// - `NumericalInstability` (`IllConditioned`) if the estimator cannot
    ///   complete.
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        self.condition_estimate_scoped(
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// Estimates the reciprocal condition number as [`Self::condition_estimate`]
    /// does, using `workspace` and `context`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::condition_estimate`].
    /// - Limit errors from the plan's options as described in
    ///   [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn condition_estimate_scoped<'scope, 'capability>(
        &self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        crate::estimate::scoped_condition(
            &self.acceptance.execution_options(),
            workspace,
            context.into(),
            || {
                Ok(crate::estimate::ConditionQuery {
                    algorithm: scioxide_core::algorithm_id!("linalg", "minimum_norm_condition", 1),
                    order: if self.rank.maximum_rank() == 0 {
                        0
                    } else {
                        self.factor.rows().max(self.factor.cols())
                    },
                    norm: self.factor.one_norm,
                    singular: !self.rank.is_full_rank(),
                })
            },
            |action, kernels| {
                let adjoint = action.adjoint;
                action.solve(
                    |values, temporary| self.inverse_ordinary(values, temporary, adjoint, kernels),
                    |rhs, work| self.inverse_scaled(rhs, work, adjoint),
                )
            },
        )
    }
    fn inverse_ordinary(
        &self,
        values: &mut [S],
        work: &mut [S],
        adjoint: bool,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let normal = !(adjoint ^ (self.factor.orientation() == HouseholderOrientation::Lq));
        let packed = self.factor.qr.packed_factors();
        let r = self.rank.rank();
        if normal {
            self.factor.qr.apply_qh_vector_in_place(values)?;
        } else {
            work[..values.len()].fill(S::ZERO);
            for (i, &source) in self.factor.permutation().iter().enumerate() {
                work[i] = values[source];
            }
            values.copy_from_slice(&work[..values.len()]);
            self.apply_z(values, work, true)?;
        }
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: crate::kernel::triangular::Storage::Dense(self.triangular()),
                triangle: scioxide_core::Triangle::Upper,
                op: if normal {
                    crate::kernel::GemmOp::None
                } else {
                    crate::kernel::GemmOp::Adjoint
                },
                diagonal: scioxide_core::DiagonalKind::Explicit,
            },
            self.maximum,
            &mut values[..r],
            r.saturating_sub(1),
            context.dispatch(),
        )?;
        values[r..].fill(S::ZERO);
        if normal {
            self.apply_z(values, work, false)?;
            work[..values.len()].copy_from_slice(values);
            values.fill(S::ZERO);
            for (i, &destination) in self.factor.permutation().iter().enumerate() {
                values[destination] = work[i];
            }
        } else {
            self.factor.qr.apply_q_vector_in_place(values)?;
            values[packed.rows()..].fill(S::ZERO);
        }
        Ok(())
    }

    fn inverse_scaled(
        &self,
        rhs: &mut [ScaledValue<S>],
        work: &mut [ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        let normal = !(adjoint ^ (self.factor.orientation() == HouseholderOrientation::Lq));
        let packed = self.factor.qr.packed_factors();
        let r = self.rank.rank();
        work.fill(ScaledValue::ZERO);
        if normal {
            work[..packed.rows()].copy_from_slice(&rhs[..packed.rows()]);
            self.factor.qr.apply_q_scaled(work, true)?;
            crate::diagnostics::matrix_substitute(
                self.triangular(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                false,
                &mut work[..r],
            )?;
            work[r..].fill(ScaledValue::ZERO);
            self.apply_z_scaled(work, false)?;
            rhs.fill(ScaledValue::ZERO);
            for (i, &original) in self.factor.permutation().iter().enumerate() {
                rhs[original] = work[i];
            }
        } else {
            for (i, &original) in self.factor.permutation().iter().enumerate() {
                work[i] = rhs[original];
            }
            self.apply_z_scaled(work, true)?;
            crate::diagnostics::matrix_substitute(
                self.triangular(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                true,
                &mut work[..r],
            )?;
            work[r..].fill(ScaledValue::ZERO);
            self.factor.qr.apply_q_scaled(work, false)?;
            rhs.fill(ScaledValue::ZERO);
            rhs[..packed.rows()].copy_from_slice(&work[..packed.rows()]);
        }
        Ok(())
    }
    fn apply_z_scaled(
        &self,
        values: &mut [ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        let (r, n) = (self.rank.rank(), self.rz.cols());
        if r == n {
            return Ok(());
        }
        for step in 0..r {
            let i = if adjoint { r - 1 - step } else { step };
            let mut dot = values[i];
            for (j, &value) in values.iter().enumerate().take(n).skip(r) {
                dot = dot.add(
                    ScaledValue::from_value(self.rz.as_slice()[i * n + j].conj())?.mul(value)?,
                )?;
            }
            dot = dot.mul(ScaledValue::from_value(if adjoint {
                self.tau[i].conj()
            } else {
                self.tau[i]
            })?)?;
            values[i] = values[i].sub(dot)?;
            for (j, value) in values.iter_mut().enumerate().take(n).skip(r) {
                *value =
                    value.sub(ScaledValue::from_value(self.rz.as_slice()[i * n + j])?.mul(dot)?)?;
            }
        }
        Ok(())
    }
}
