//! Primal active-set solver for the QP and linear least-squares subproblems of SQP.
use super::{SqpState, invalid, resize};
use crate::context::SolveSession;
use crate::{DenseKktProvider, KktProvider, OptimalityTolerances, OptimizeTermination};
use alloc::vec::Vec;
use scioxide_core::Argument;
use scioxide_core::{
    Budget, EvaluationRole, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, MatrixView,
    RankPolicy, Vector,
};
use scioxide_sparse::LdlInertia;

/// Selects the convex objective of an [`SqpSubproblem`] in the step `d`.
#[derive(Clone, Copy)]
pub enum SqpSubproblemObjective<'a, S: RealScalar + NumericScalar<Real = S>> {
    /// Minimizes `dᵀ H d / 2 + gᵀ d` with a symmetric positive definite `H`.
    Quadratic {
        /// The symmetric matrix `H`.
        hessian: MatrixView<'a, S>,
        /// The vector `g`.
        gradient: &'a [S],
    },
    /// Minimizes `‖A d − b‖² / 2`. `A` may be rectangular or rank deficient;
    /// the normal matrix `AᵀA` is not formed.
    LeastSquares {
        /// The matrix `A`.
        design: MatrixView<'a, S>,
        /// The vector `b`.
        target: &'a [S],
    },
}
/// Describes one QP or linear least-squares subproblem of [`Sqp`](crate::Sqp).
///
/// ```text
/// minimize φ(d)  subject to  cᵢ + Jᵢ d = 0  (i < equalities),
///                            cᵢ + Jᵢ d ≤ 0  (i ≥ equalities),
///                            lⱼ ≤ dⱼ ≤ uⱼ
/// ```
///
/// where `φ` is the [`objective`](Self::objective), `c` are the
/// [`values`](Self::values), `J` is the [`jacobian`](Self::jacobian) and the
/// step bounds `[lⱼ, uⱼ]` contain zero.
#[derive(Clone, Copy)]
pub struct SqpSubproblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    /// The objective `φ`.
    pub objective: SqpSubproblemObjective<'a, S>,
    /// The Jacobian `J` of the nonlinear rows; bounds have no rows.
    pub jacobian: MatrixView<'a, S>,
    /// The constraint values `c` at the linearization point.
    pub values: &'a [S],
    /// The number of leading equality rows.
    pub equalities: usize,
    /// The step bounds `(lⱼ, uⱼ)`, one pair per variable; `None` is unbounded.
    pub bounds: &'a [(Option<S>, Option<S>)],
    /// The feasibility, optimality and rank tolerances.
    pub tolerances: OptimalityTolerances<S>,
    /// The iteration limit of the subproblem solver.
    pub max_iterations: Budget,
}
/// Reports the outcome of an [`SqpSubproblemProvider`] solve.
///
/// `SuspectedInfeasible` means the linearized constraints are inconsistent. It
/// does not mean that the nonlinear problem is infeasible.
#[derive(Debug, Clone, Copy)]
pub struct SqpSubproblemReport {
    pub(crate) termination: OptimizeTermination,
    pub(crate) active_rank: usize,
    pub(crate) multipliers_ambiguous: bool,
    pub(crate) inertia: Option<LdlInertia>,
}
impl SqpSubproblemReport {
    /// Creates a report with status `termination`, full rank, unique
    /// multipliers and no inertia.
    ///
    /// Use `Converged` for a solved subproblem, `SuspectedInfeasible` for
    /// inconsistent constraints, or the status of an unsuccessful stop. Do not
    /// use `GradientSmall`: a small objective gradient does not satisfy the
    /// KKT conditions of a constrained problem.
    pub const fn new(termination: OptimizeTermination) -> Self {
        Self {
            termination,
            active_rank: 0,
            multipliers_ambiguous: false,
            inertia: None,
        }
    }
    /// Sets the numerical rank of the active constraint gradients.
    #[must_use]
    pub const fn with_active_rank(mut self, rank: usize) -> Self {
        self.active_rank = rank;
        self
    }
    /// Marks the active constraint gradients as linearly dependent, so the
    /// multipliers may not be unique.
    #[must_use]
    pub const fn with_ambiguous_multipliers(mut self, ambiguous: bool) -> Self {
        self.multipliers_ambiguous = ambiguous;
        self
    }
    /// Sets the inertia of the last symmetric KKT matrix factored.
    #[must_use]
    pub const fn with_inertia(mut self, inertia: LdlInertia) -> Self {
        self.inertia = Some(inertia);
        self
    }
    /// Returns the termination status.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the numerical rank of the active constraint gradients.
    pub const fn active_rank(&self) -> usize {
        self.active_rank
    }
    /// Returns `true` if the active constraint gradients were linearly
    /// dependent.
    pub const fn multipliers_ambiguous(&self) -> bool {
        self.multipliers_ambiguous
    }
    /// Returns the inertia of the last KKT matrix factored, if any.
    pub const fn inertia(&self) -> Option<LdlInertia> {
        self.inertia
    }
}
/// Solves the QP subproblems of [`Sqp`](crate::Sqp).
///
/// Implement this trait to replace [`DenseSqpSubproblem`], and install it with
/// [`SqpWorkspace::with_subproblem_provider`](crate::SqpWorkspace::with_subproblem_provider).
/// An implementation counts its iterations in `context`, stops at
/// [`SqpSubproblem::max_iterations`] or the limits of `context`, and writes
/// finite values. [`Sqp`](crate::Sqp) checks each step that is reported as
/// `Converged` against the linearized constraints before using it. The `Debug`
/// output should describe the provider's configuration.
pub trait SqpSubproblemProvider<S: RealScalar + NumericScalar<Real = S>>: core::fmt::Debug {
    /// Solves `problem` and writes the step, multipliers and active set to
    /// `state`.
    ///
    /// # Errors
    ///
    /// Implementations return an error for invalid input or a numerical
    /// failure. A subproblem that cannot be solved within its limits, or whose
    /// constraints are inconsistent, is reported through the returned status.
    fn solve_into(
        &mut self,
        problem: SqpSubproblem<'_, S>,
        state: &mut SqpState<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SqpSubproblemReport, SciOxideError>;
}

/// Solves SQP subproblems with a dense primal active-set method.
///
/// A Householder QR factorization selects linearly independent active
/// constraint gradients. Each equality-constrained KKT system is solved with
/// [`DenseKktProvider`], and singular least-squares systems with a
/// minimum-norm SVD solution. A feasible starting point comes from an elastic
/// phase that minimizes the constraint violation with one extra variable.
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §16.5.
pub struct DenseSqpSubproblem<S: RealScalar + NumericScalar<Real = S>> {
    n: usize,
    m: usize,
    residuals: usize,
    a: Matrix<S>,
    b: Vec<S>,
    eq: Vec<bool>,
    scale: Vec<S>,
    source: Vec<usize>,
    x: Vec<S>,
    p: Vec<S>,
    gradient: Vec<S>,
    active: Vec<usize>,
    independent: Vec<usize>,
    preferred: Vec<usize>,
    transpose: Matrix<S>,
    qr: HouseholderDecomposition<S>,
    qw: LinalgWorkspace<'static, S>,
    kkt: Matrix<S>,
    rhs: Vector<S>,
    solution: Vector<S>,
    scratch: Vector<S>,
    factor: DenseKktProvider<S>,
    lambda: Vec<S>,
    report: SqpSubproblemReport,
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for DenseSqpSubproblem<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DenseSqpSubproblem")
            .field("variables", &self.n)
            .field("constraints", &self.m)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> DenseSqpSubproblem<S> {
    /// Allocates a solver for up to `variables` variables and `constraints`
    /// nonlinear rows.
    ///
    /// `residuals` is the largest number of rows of a least-squares objective,
    /// or zero if only quadratic objectives are solved.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(
        variables: usize,
        constraints: usize,
        residuals: usize,
    ) -> Result<Self, SciOxideError> {
        let n = scioxide_core::backend::checked_add(
            variables,
            1,
            scioxide_core::quantity!("sqp_elastic_dimension"),
        )?;
        let rows = scioxide_core::backend::checked_product(
            &[
                2,
                scioxide_core::backend::checked_add(
                    constraints,
                    n,
                    scioxide_core::quantity!("sqp_rows"),
                )?,
            ],
            scioxide_core::quantity!("sqp_rows"),
        )?;
        let order = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_product(&[2, n], scioxide_core::quantity!("sqp_kkt"))?,
            residuals,
            scioxide_core::quantity!("sqp_kkt"),
        )?;
        let mut w = Self {
            n: variables,
            m: constraints,
            residuals,
            a: Matrix::empty(),
            b: Vec::new(),
            eq: Vec::new(),
            scale: Vec::new(),
            source: Vec::new(),
            x: Vec::new(),
            p: Vec::new(),
            gradient: Vec::new(),
            active: Vec::new(),
            independent: Vec::new(),
            preferred: Vec::new(),
            transpose: Matrix::empty(),
            qr: HouseholderDecomposition::empty(),
            qw: LinalgWorkspace::new(),
            kkt: Matrix::empty(),
            rhs: Vector::empty(),
            solution: Vector::empty(),
            scratch: Vector::empty(),
            factor: DenseKktProvider::new(order)?,
            lambda: Vec::new(),
            report: SqpSubproblemReport {
                termination: OptimizeTermination::Stagnation,
                active_rank: 0,
                multipliers_ambiguous: false,
                inertia: None,
            },
        };
        scioxide_core::backend::try_reserve(
            &mut w.preferred,
            rows,
            scioxide_core::quantity!("sqp_preferred_active_set"),
        )?;
        w.a.try_resize_flat(rows, n)?;
        w.transpose.try_resize_flat(n, rows)?;
        w.kkt.try_resize_flat(order, order)?;
        for v in [&mut w.x, &mut w.p, &mut w.gradient] {
            resize(v, n, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        }
        for v in [&mut w.b, &mut w.scale, &mut w.lambda] {
            resize(v, rows, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        }
        resize(&mut w.eq, rows, false)?;
        resize(&mut w.source, rows, 0)?;
        scioxide_core::backend::try_reserve(
            &mut w.active,
            rows,
            scioxide_core::quantity!("sqp_active_set"),
        )?;
        scioxide_core::backend::try_reserve(
            &mut w.independent,
            rows,
            scioxide_core::quantity!("sqp_independent_set"),
        )?;
        for v in [&mut w.rhs, &mut w.solution, &mut w.scratch] {
            v.try_resize(order, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        }
        if residuals > 0 {
            w.factor.reserve_least_squares_fallback()?;
        }
        w.qr.try_reserve(n, rows, HouseholderOrientation::Qr)?;
        w.qw.try_reserve(
            &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::factor(
                    n,
                    rows,
                    HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?,
        )?;
        Ok(w)
    }
    pub(super) fn dimensions(&self) -> (usize, usize, usize) {
        (self.n, self.m, self.residuals)
    }
    fn validate(&self, model: SqpSubproblem<'_, S>) -> Result<(), SciOxideError> {
        let n = model.bounds.len();
        let m = model.values.len();
        if n != self.n
            || m > self.m
            || model.jacobian.rows() != m
            || model.jacobian.cols() != n
            || model.equalities > m
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "subproblem dimensions exceed prepared shape",
            ));
        }
        for &(lo, hi) in model.bounds {
            if lo.is_some_and(|v| !v.is_finite() || v > <_ as scioxide_core::AlgebraicScalar>::ZERO)
                || hi.is_some_and(|v| {
                    !v.is_finite() || v < <_ as scioxide_core::AlgebraicScalar>::ZERO
                })
            {
                return Err(invalid(
                    scioxide_core::argument!("problem.bounds"),
                    "step bounds must be finite and contain zero",
                ));
            }
        }
        crate::utils::validate_finite(model.values)?;
        for i in 0..m {
            for j in 0..n {
                crate::utils::validate_finite(&[model.jacobian.get(i, j)?])?;
            }
        }
        match model.objective {
            SqpSubproblemObjective::Quadratic { hessian, gradient } => {
                if hessian.rows() != n || hessian.cols() != n || gradient.len() != n {
                    return Err(invalid(
                        scioxide_core::argument!("problem.objective"),
                        "quadratic model shape",
                    ));
                }
                crate::utils::validate_finite(gradient)?;
                for i in 0..n {
                    for j in 0..n {
                        let a = hessian.get(i, j)?;
                        if !a.is_finite() || a != hessian.get(j, i)? {
                            return Err(invalid(
                                scioxide_core::argument!("problem.objective"),
                                "SQP Hessian must be finite and exactly symmetric",
                            ));
                        }
                    }
                }
            }
            SqpSubproblemObjective::LeastSquares { design, target } => {
                if design.cols() != n
                    || design.rows() != target.len()
                    || target.len() > self.residuals
                {
                    return Err(invalid(
                        scioxide_core::argument!("problem.objective"),
                        "least-squares model shape",
                    ));
                }
                crate::utils::validate_finite(target)?;
                for i in 0..design.rows() {
                    for j in 0..n {
                        crate::utils::validate_finite(&[design.get(i, j)?])?;
                    }
                }
            }
        }
        Ok(())
    }
    fn rows(&mut self, model: SqpSubproblem<'_, S>, elastic: bool) -> Result<usize, SciOxideError> {
        let n = self.n + usize::from(elastic);
        let m = model.values.len();
        let count = m
            + if elastic { model.equalities + 1 } else { 0 }
            + model
                .bounds
                .iter()
                .map(|&(l, u)| {
                    if l.is_some() && l == u {
                        1
                    } else {
                        usize::from(l.is_some()) + usize::from(u.is_some())
                    }
                })
                .sum::<usize>();
        self.a.try_resize_flat(count, n)?;
        self.a
            .as_mut_slice()
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        let mut row = 0;
        for i in 0..m {
            for sign in [<_ as scioxide_core::AlgebraicScalar>::ONE, -S::ONE]
                .into_iter()
                .take(if elastic && i < model.equalities {
                    2
                } else {
                    1
                })
            {
                let scale = (0..self.n).try_fold(S::ZERO, |s, j| {
                    Ok::<_, SciOxideError>(s.max(NumericScalar::abs(model.jacobian.get(i, j)?)))
                })?;
                let scale = if scale == <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    <_ as scioxide_core::AlgebraicScalar>::ONE
                } else {
                    scale
                };
                for j in 0..self.n {
                    self.a.as_mut_slice()[row * n + j] = sign * model.jacobian.get(i, j)? / scale;
                }
                if elastic {
                    self.a.as_mut_slice()[row * n + self.n] = -S::ONE;
                }
                self.b[row] = -sign * model.values[i] / scale;
                self.scale[row] = scale;
                self.eq[row] = !elastic && i < model.equalities;
                self.source[row] = i;
                row += 1;
            }
        }
        for (j, &(lo, hi)) in model.bounds.iter().enumerate() {
            if lo.is_some() && lo == hi {
                self.a.as_mut_slice()[row * n + j] = <_ as scioxide_core::AlgebraicScalar>::ONE;
                self.b[row] = lo.unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                self.eq[row] = true;
                self.scale[row] = <_ as scioxide_core::AlgebraicScalar>::ONE;
                self.source[row] = m + 2 * j + 1;
                row += 1;
            } else {
                for (endpoint, sign, id) in [
                    (lo, -S::ONE, m + 2 * j),
                    (
                        hi,
                        <_ as scioxide_core::AlgebraicScalar>::ONE,
                        m + 2 * j + 1,
                    ),
                ] {
                    if let Some(endpoint) = endpoint {
                        self.a.as_mut_slice()[row * n + j] = sign;
                        self.b[row] = sign * endpoint;
                        self.eq[row] = false;
                        self.scale[row] = <_ as scioxide_core::AlgebraicScalar>::ONE;
                        self.source[row] = id;
                        row += 1;
                    }
                }
            }
        }
        if elastic {
            self.a.as_mut_slice()[row * n + self.n] = -S::ONE;
            self.b[row] = <_ as scioxide_core::AlgebraicScalar>::ZERO;
            self.eq[row] = false;
            self.scale[row] = <_ as scioxide_core::AlgebraicScalar>::ONE;
            self.source[row] = usize::MAX;
            row += 1;
        }
        Ok(row)
    }
    fn rank(
        &mut self,
        n: usize,
        model: SqpSubproblem<'_, S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let m = self.active.len();
        self.transpose.try_resize_flat(n, m)?;
        for (j, &row) in self.active.iter().enumerate() {
            for i in 0..n {
                self.transpose.as_mut_slice()[i * m + j] = self.a.as_slice()[row * n + i];
            }
        }
        session.invoke(EvaluationRole::Factorization, || {
            HouseholderDecomposition::factor_into(
                &self.transpose,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut self.qr,
                &mut self.qw,
                session.execution()?,
            )
        })?;
        let rank = self
            .qr
            .analyze_rank(RankPolicy::Threshold(model.tolerances.rank()))?
            .rank();
        self.independent.clear();
        self.independent.extend(
            self.qr
                .permutation()
                .iter()
                .take(rank)
                .map(|&j| self.active[j]),
        );
        self.report.active_rank = rank;
        self.report.multipliers_ambiguous = m > rank;
        Ok(())
    }
    fn direction(
        &mut self,
        model: SqpSubproblem<'_, S>,
        elastic: bool,
        penalty: S,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.n + usize::from(elastic);
        self.rank(n, model, session)?;
        let r = if elastic {
            0
        } else {
            match model.objective {
                SqpSubproblemObjective::LeastSquares { design, .. } => design.rows(),
                _ => 0,
            }
        };
        let least_squares =
            !elastic && matches!(model.objective, SqpSubproblemObjective::LeastSquares { .. });
        let rank = self.independent.len();
        let order = r + n + rank;
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut self.kkt, order)?;
        if elastic {
            assembly.diagonal(0, n, |_| Ok(<S as scioxide_core::AlgebraicScalar>::ONE))?;
        } else {
            match model.objective {
                SqpSubproblemObjective::Quadratic { hessian, .. } => {
                    assembly.symmetric(0, n, |i, j| hessian.get(i, j))?
                }
                SqpSubproblemObjective::LeastSquares { design, .. } => {
                    assembly.diagonal(0, r, |_| Ok(<S as scioxide_core::AlgebraicScalar>::ONE))?;
                    assembly.coupling(0, r, r, n, |i, j| design.get(i, j))?;
                }
            }
        }
        let sign = if least_squares {
            -S::ONE
        } else {
            <S as scioxide_core::AlgebraicScalar>::ONE
        };
        assembly.coupling(r + n, r, rank, n, |i, j| {
            Ok(sign * self.a.as_slice()[self.independent[i] * n + j])
        })?;
        self.rhs
            .try_resize(order, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.rhs
            .as_mut_slice()
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        self.solution
            .try_resize(order, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        if elastic {
            for i in 0..n {
                self.gradient[i] = self.x[i]
                    + if i == self.n {
                        penalty
                    } else {
                        <_ as scioxide_core::AlgebraicScalar>::ZERO
                    };
                self.rhs.as_mut_slice()[i] = -self.gradient[i];
            }
        } else {
            match model.objective {
                SqpSubproblemObjective::Quadratic { hessian, gradient } => {
                    for (i, &initial_gradient) in gradient.iter().enumerate().take(n) {
                        let mut sum = scioxide_core::ProductSum::new();
                        sum.add_product(
                            initial_gradient,
                            <_ as scioxide_core::AlgebraicScalar>::ONE,
                        )?;
                        for j in 0..n {
                            let h = hessian.get(i, j)?;
                            sum.add_product(h, self.x[j])?;
                        }
                        self.gradient[i] = sum.finish()?;
                        self.rhs.as_mut_slice()[i] = -self.gradient[i];
                    }
                }
                SqpSubproblemObjective::LeastSquares { design, target } => {
                    self.gradient[..n].fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                    for (i, &b) in target.iter().enumerate() {
                        let mut residual = scioxide_core::ProductSum::new();
                        residual.add_product(-b, <_ as scioxide_core::AlgebraicScalar>::ONE)?;
                        for j in 0..n {
                            let a = design.get(i, j)?;
                            residual.add_product(a, self.x[j])?;
                        }
                        let v = residual.finish()?;
                        self.rhs.as_mut_slice()[i] = -v;
                        for j in 0..n {
                            self.gradient[j] = crate::utils::trial_component(
                                self.gradient[j],
                                v,
                                design.get(i, j)?,
                            )?;
                        }
                    }
                }
            }
        }
        let sign = if least_squares {
            -S::ONE
        } else {
            <_ as scioxide_core::AlgebraicScalar>::ONE
        };
        for (i, &row) in self.independent.iter().enumerate() {
            self.rhs.as_mut_slice()[r + n + i] = sign
                * (self.b[row]
                    - crate::utils::dot(&self.a.as_slice()[row * n..(row + 1) * n], &self.x[..n])?);
        }
        self.factor.use_least_squares_rank_policy(
            least_squares.then_some(RankPolicy::Threshold(model.tolerances.rank())),
        );
        let inertia = session.invoke(EvaluationRole::Factorization, || {
            self.factor.factor(self.kkt.view(), session.execution()?)
        })?;
        self.report.inertia = Some(inertia);
        if !least_squares && !crate::kkt::matches_inertia(inertia, n, rank)? {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "quadratic KKT inertia differs from positive-definite reduced model",
            ));
        }
        session.invoke(EvaluationRole::LinearSolve, || {
            self.factor.solve_into(
                self.rhs.as_slice(),
                self.solution.as_mut_slice(),
                session.execution()?,
            )
        })?;
        let scale = self
            .rhs
            .as_slice()
            .iter()
            .chain(self.solution.as_slice())
            .fold(S::ONE, |a, &v| a.max(NumericScalar::abs(v)));
        let floor =
            (S::EPSILON * S::try_from_usize_exact(order)? * scale * S::try_from_f64_lossy(16.0)?)
                .min(
                    model
                        .tolerances
                        .gradient()
                        .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?
                        .min(
                            model
                                .tolerances
                                .feasibility()
                                .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?,
                        ),
                );
        crate::kkt::verify_residual_with_floor(
            self.kkt.view(),
            self.rhs.as_slice(),
            self.solution.as_slice(),
            floor,
        )?;
        self.p[..n].copy_from_slice(&self.solution.as_slice()[r..r + n]);
        self.lambda
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        for (i, &row) in self.independent.iter().enumerate() {
            self.lambda[row] = self.solution.as_slice()[r + n + i];
        }
        Ok(())
    }
    fn stationarity(&mut self, n: usize) -> Result<S, SciOxideError> {
        self.scratch
            .try_resize(n, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        for j in 0..n {
            let mut sum = scioxide_core::ProductSum::new();
            sum.add_product(self.gradient[j], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
            for &row in &self.independent {
                sum.add_product(self.a.as_slice()[row * n + j], self.lambda[row])?;
            }
            self.scratch.as_mut_slice()[j] = sum.finish()?;
        }
        crate::utils::norm2(self.scratch.as_slice())
    }
    fn active_solve(
        &mut self,
        model: SqpSubproblem<'_, S>,
        elastic: bool,
        penalty: S,
        remaining: &mut usize,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let n = self.n + usize::from(elastic);
        let rows = self.a.rows();
        let tol = model
            .tolerances
            .feasibility()
            .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
        self.active.clear();
        for row in 0..rows {
            let v = crate::utils::dot(&self.a.as_slice()[row * n..(row + 1) * n], &self.x[..n])?;
            if self.eq[row] || NumericScalar::abs(v - self.b[row]) <= tol {
                self.active.push(row);
            }
        }
        self.active.sort_unstable_by_key(|&row| {
            (
                self.preferred
                    .iter()
                    .position(|&id| id == self.source[row])
                    .unwrap_or(usize::MAX),
                row,
            )
        });
        while *remaining > 0 {
            *remaining -= 1;
            if let Some(stop) = session.trial_iteration()? {
                return Ok(stop);
            }
            self.direction(model, elastic, penalty, session)?;
            // Displacements and stationarity have different units. Large curvature
            // can require a tiny displacement even with a large KKT residual.
            if self.stationarity(n)?
                <= model
                    .tolerances
                    .gradient()
                    .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?
            {
                let dual_target = model
                    .tolerances
                    .dual_feasibility()
                    .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
                let leaving = self
                    .independent
                    .iter()
                    .copied()
                    .filter(|&row| !self.eq[row] && self.lambda[row] < -dual_target)
                    .min_by(|&a, &b| {
                        self.lambda[a]
                            .partial_cmp(&self.lambda[b])
                            .unwrap_or(core::cmp::Ordering::Equal)
                    });
                if let Some(row) = leaving {
                    self.active.retain(|&i| i != row);
                    continue;
                }
                return Ok(OptimizeTermination::Converged);
            }
            let mut alpha = <_ as scioxide_core::AlgebraicScalar>::ONE;
            let mut blocker = None;
            for row in 0..rows {
                if self.eq[row] || self.active.contains(&row) {
                    continue;
                }
                let a = &self.a.as_slice()[row * n..(row + 1) * n];
                let slope = crate::utils::dot(a, &self.p[..n])?;
                if slope > <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    let distance = ((self.b[row] - crate::utils::dot(a, &self.x[..n])?) / slope)
                        .max(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                    if distance < alpha {
                        alpha = distance;
                        blocker = Some(row);
                    }
                }
            }
            session.proposal()?;
            let mut moved = false;
            for i in 0..n {
                let next = crate::utils::trial_component(self.x[i], alpha, self.p[i])?;
                moved |= next != self.x[i];
                self.x[i] = next;
            }
            if !moved && blocker.is_none() {
                session.reject()?;
                return Ok(OptimizeTermination::Stagnation);
            }
            session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
            if let Some(row) = blocker {
                self.active.push(row);
            }
        }
        Ok(OptimizeTermination::IterationLimit)
    }
    fn consistent_equalities(
        &mut self,
        model: SqpSubproblem<'_, S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        if model.equalities == 0 {
            return Ok(true);
        }
        let mut rank = 0;
        for augmented in [false, true] {
            let rows = self.n + usize::from(augmented);
            self.transpose.try_resize_flat(rows, model.equalities)?;
            for j in 0..model.equalities {
                let scale = (0..self.n).try_fold(S::ZERO, |a, i| {
                    Ok::<_, SciOxideError>(a.max(NumericScalar::abs(model.jacobian.get(j, i)?)))
                })?;
                let scale = if scale == <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    <_ as scioxide_core::AlgebraicScalar>::ONE
                } else {
                    scale
                };
                for i in 0..self.n {
                    self.transpose.as_mut_slice()[i * model.equalities + j] =
                        model.jacobian.get(j, i)? / scale;
                }
                if augmented {
                    self.transpose.as_mut_slice()[self.n * model.equalities + j] =
                        model.values[j] / scale;
                }
            }
            session.invoke(EvaluationRole::Factorization, || {
                HouseholderDecomposition::factor_into(
                    &self.transpose,
                    HouseholderOrientation::Qr,
                    scioxide_linalg::HouseholderPivoting::Column,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut self.qr,
                    &mut self.qw,
                    session.execution()?,
                )
            })?;
            let observed = self
                .qr
                .analyze_rank(RankPolicy::Threshold(model.tolerances.rank()))?
                .rank();
            if augmented {
                return Ok(observed <= rank);
            }
            rank = observed;
        }
        Ok(true)
    }
    fn solve(
        &mut self,
        model: SqpSubproblem<'_, S>,
        state: &mut SqpState<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<SqpSubproblemReport, SciOxideError> {
        self.validate(model)?;
        self.preferred.clear();
        let ids = model.values.len() + 2 * self.n;
        if state.active.iter().any(|&id| id >= ids) {
            return Err(invalid(
                scioxide_core::argument!("state"),
                "warm active-set index is outside the problem",
            ));
        }
        self.preferred.extend_from_slice(&state.active);
        state.reserve(self.n, model.values.len())?;
        self.report = SqpSubproblemReport {
            termination: OptimizeTermination::Stagnation,
            active_rank: 0,
            multipliers_ambiguous: false,
            inertia: None,
        };
        self.factor.use_rank_policy(None);
        if let SqpSubproblemObjective::Quadratic { hessian, .. } = model.objective
            && !hessian.is_positive_diagonally_dominant()?
        {
            let inertia = session.invoke(EvaluationRole::Factorization, || {
                self.factor.factor(hessian, session.execution()?)
            })?;
            if inertia.negative() != 0 || inertia.zero() != 0 {
                return Err(invalid(
                    scioxide_core::argument!("problem.objective"),
                    "SQP quadratic models require a positive-definite Hessian",
                ));
            }
        }
        if !self.consistent_equalities(model, session)? {
            self.report.termination = OptimizeTermination::SuspectedInfeasible;
            return Ok(self.report);
        }
        self.x.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        let mut remaining = model.max_iterations.get();
        let tol = model
            .tolerances
            .feasibility()
            .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
        let needs_phase = model.values.iter().enumerate().any(|(i, &v)| {
            if i < model.equalities {
                NumericScalar::abs(v) > tol
            } else {
                v > tol
            }
        });
        // First solve the equality-constrained model. When it satisfies every
        // inequality and bound, its independently checked KKT system is already
        // the complete convex subproblem solution; no elastic phase is needed.
        self.rows(model, false)?;
        self.active.clear();
        self.active
            .extend((0..self.a.rows()).filter(|&row| self.eq[row]));
        if let Some(stop) = session.trial_iteration()? {
            self.report.termination = stop;
            return Ok(self.report);
        }
        remaining -= 1;
        self.direction(
            model,
            false,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
            session,
        )?;
        let mut direct = true;
        for row in 0..self.a.rows() {
            let residual = (crate::utils::dot(
                &self.a.as_slice()[row * self.n..(row + 1) * self.n],
                &self.p[..self.n],
            )? - self.b[row])
                * self.scale[row];
            direct &= (if self.eq[row] {
                NumericScalar::abs(residual)
            } else {
                residual
            }) <= tol;
        }
        if direct {
            self.x[..self.n].copy_from_slice(&self.p[..self.n]);
        }
        if !direct && needs_phase {
            let rows = self.rows(model, true)?;
            self.x[self.n] = self.b[..rows].iter().fold(S::ZERO, |v, &b| v.max(-b))
                + <_ as scioxide_core::AlgebraicScalar>::ONE;
            let mut penalty = S::TEN;
            for _ in 0..8 {
                let status = self.active_solve(model, true, penalty, &mut remaining, session)?;
                if status != OptimizeTermination::Converged {
                    self.report.termination = status;
                    return Ok(self.report);
                }
                if self.x[self.n] <= tol {
                    break;
                }
                penalty = scioxide_core::checked_float_product(penalty, S::TEN)?;
            }
            if self.x[self.n] > tol {
                self.report.termination = OptimizeTermination::SuspectedInfeasible;
                state.step.copy_from_slice(&self.x[..self.n]);
                return Ok(self.report);
            }
        }
        self.report.termination = if direct {
            OptimizeTermination::Converged
        } else {
            self.rows(model, false)?;
            self.active_solve(
                model,
                false,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                &mut remaining,
                session,
            )?
        };
        if self.report.termination != OptimizeTermination::Converged {
            return Ok(self.report);
        }
        state.step.copy_from_slice(&self.x[..self.n]);
        state
            .multipliers
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        state
            .lower
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        state
            .upper
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        state.active.clear();
        for row in 0..self.a.rows() {
            let id = self.source[row];
            let lambda = self.lambda[row] / self.scale[row];
            if id < model.values.len() {
                state.multipliers[id] = lambda;
            } else {
                let j = (id - model.values.len()) / 2;
                if self.eq[row] {
                    state.lower[j] = (-lambda).max(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                    state.upper[j] = lambda.max(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                } else if (id - model.values.len()).is_multiple_of(2) {
                    state.lower[j] = lambda;
                } else {
                    state.upper[j] = lambda;
                }
            }
            if self.active.contains(&row) {
                state.active.push(id);
            }
        }
        if !verify(model, state, scioxide_core::argument!("state"))? {
            self.report.termination = OptimizeTermination::Stagnation;
        }
        Ok(self.report)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SqpSubproblemProvider<S> for DenseSqpSubproblem<S> {
    fn solve_into(
        &mut self,
        problem: SqpSubproblem<'_, S>,
        state: &mut SqpState<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SqpSubproblemReport, SciOxideError> {
        let session = SolveSession::new(context, scioxide_core::argument!("context"));
        let outcome = self.solve(problem, state, &session);
        match outcome {
            Err(error) => {
                if let Some(stop) = session.stopped() {
                    self.report.termination = stop;
                    Ok(self.report)
                } else {
                    Err(error)
                }
            }
            Ok(report) => Ok(report),
        }
    }
}
/// Independent original-model residual check, also applied to injected providers.
pub(super) fn verify<S: RealScalar + NumericScalar<Real = S>>(
    model: SqpSubproblem<'_, S>,
    state: &SqpState<S>,
    argument: Argument,
) -> Result<bool, SciOxideError> {
    let n = model.bounds.len();
    let m = model.values.len();
    if state.step.len() != n
        || state.multipliers.len() != m
        || state.lower.len() != n
        || state.upper.len() != n
    {
        return Err(invalid(argument, "subproblem state shape"));
    }
    if state.active.iter().any(|&id| id >= m + 2 * n) {
        return Err(invalid(
            argument,
            "provider active-set index is outside the problem",
        ));
    }
    for v in [&state.step, &state.multipliers, &state.lower, &state.upper] {
        crate::utils::validate_finite(v)?;
    }
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let primal_scale = state
        .step
        .iter()
        .chain(model.values)
        .fold(one, |a, &v| a.max(NumericScalar::abs(v)));
    let dual_scale = state
        .multipliers
        .iter()
        .chain(&state.lower)
        .chain(&state.upper)
        .fold(one, |a, &v| a.max(NumericScalar::abs(v)));
    // Bound cancellation in each component of g + Hs + J^T lambda + z.
    // The same reference has gradient units for every subproblem objective.
    let mut gradient_scale = one;
    for j in 0..n {
        let mut magnitude = scioxide_core::ProductSum::new();
        match model.objective {
            SqpSubproblemObjective::Quadratic { hessian, gradient } => {
                magnitude.add_product(NumericScalar::abs(gradient[j]), one)?;
                for k in 0..n {
                    magnitude.add_product(
                        NumericScalar::abs(hessian.get(j, k)?),
                        NumericScalar::abs(state.step[k]),
                    )?;
                }
            }
            SqpSubproblemObjective::LeastSquares { design, target } => {
                for (i, &b) in target.iter().enumerate() {
                    let mut residual = scioxide_core::ProductSum::new();
                    residual.add_product(-b, one)?;
                    for k in 0..n {
                        residual.add_product(design.get(i, k)?, state.step[k])?;
                    }
                    magnitude.add_product(
                        NumericScalar::abs(design.get(i, j)?),
                        NumericScalar::abs(residual.finish()?),
                    )?;
                }
            }
        }
        for i in 0..m {
            magnitude.add_product(
                NumericScalar::abs(model.jacobian.get(i, j)?),
                NumericScalar::abs(state.multipliers[i]),
            )?;
        }
        magnitude.add_product(NumericScalar::abs(state.lower[j]), one)?;
        magnitude.add_product(NumericScalar::abs(state.upper[j]), one)?;
        gradient_scale = gradient_scale.max(magnitude.finish()?);
    }
    let feasible = model.tolerances.feasibility().threshold(primal_scale)?;
    let dual = model.tolerances.dual_feasibility().threshold(dual_scale)?;
    let comp =
        model
            .tolerances
            .complementarity()
            .threshold(scioxide_core::checked_float_product(
                primal_scale,
                gradient_scale,
            )?)?;
    let stationarity = model.tolerances.gradient().threshold(gradient_scale)?;
    for i in 0..m {
        let mut sum = scioxide_core::ProductSum::new();
        sum.add_product(model.values[i], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
        for j in 0..n {
            sum.add_product(model.jacobian.get(i, j)?, state.step[j])?;
        }
        let v = sum.finish()?;
        if (if i < model.equalities {
            NumericScalar::abs(v)
        } else {
            v
        }) > feasible
        {
            return Ok(false);
        }
        if i >= model.equalities
            && (state.multipliers[i] < -dual || NumericScalar::abs(state.multipliers[i] * v) > comp)
        {
            return Ok(false);
        }
    }
    for (j, &(lo, hi)) in model.bounds.iter().enumerate() {
        if state.lower[j] < -dual || state.upper[j] < -dual {
            return Ok(false);
        }
        for (endpoint, lambda, sign) in [
            (lo, state.lower[j], -S::ONE),
            (
                hi,
                state.upper[j],
                <_ as scioxide_core::AlgebraicScalar>::ONE,
            ),
        ] {
            match endpoint {
                Some(b) => {
                    let v = sign * (state.step[j] - b);
                    if v > feasible || NumericScalar::abs(lambda * v) > comp {
                        return Ok(false);
                    }
                }
                None => {
                    if lambda != <_ as scioxide_core::AlgebraicScalar>::ZERO {
                        return Err(invalid(argument, "multiplier on absent bound"));
                    }
                }
            }
        }
        let mut sum = scioxide_core::ProductSum::new();
        match model.objective {
            SqpSubproblemObjective::Quadratic { hessian, gradient } => {
                sum.add_product(gradient[j], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
                for k in 0..n {
                    sum.add_product(hessian.get(j, k)?, state.step[k])?;
                }
            }
            SqpSubproblemObjective::LeastSquares { design, target } => {
                for (i, &b) in target.iter().enumerate() {
                    let mut r = scioxide_core::ProductSum::new();
                    r.add_product(-b, <_ as scioxide_core::AlgebraicScalar>::ONE)?;
                    for k in 0..n {
                        r.add_product(design.get(i, k)?, state.step[k])?;
                    }
                    sum.add_product(design.get(i, j)?, r.finish()?)?;
                }
            }
        }
        for i in 0..m {
            sum.add_product(model.jacobian.get(i, j)?, state.multipliers[i])?;
        }
        sum.add_product(state.upper[j], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
        sum.add_product(-state.lower[j], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
        if NumericScalar::abs(sum.finish()?) > stationarity {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod residual_scale_tests {
    use super::*;
    #[test]
    fn finite_subproblem_residuals_use_model_and_multiplier_magnitudes() {
        let hessian = Matrix::from_vec(1, 1, alloc::vec![1e12_f64]).unwrap();
        let rows = Matrix::zeros(0, 1).unwrap();
        let bounds = [(None, None)];
        let model = SqpSubproblem {
            objective: SqpSubproblemObjective::Quadratic {
                hessian: hessian.view(),
                gradient: &[1e12],
            },
            jacobian: rows.view(),
            values: &[],
            equalities: 0,
            bounds: &bounds,
            tolerances: OptimalityTolerances::new(
                Default::default(),
                scioxide_core::GradientTolerance::new(0.0, 1e-10).unwrap(),
                Default::default(),
                Default::default(),
            ),
            max_iterations: Budget::ONE,
        };
        let mut state = SqpState::new();
        state.reserve(1, 0).unwrap();
        state.step[0] = -1.0 + 5e-13;
        assert!(verify(model, &state, scioxide_core::argument!("state")).unwrap());
        state.step[0] = -1.0 + 1e-5;
        assert!(!verify(model, &state, scioxide_core::argument!("state")).unwrap());
    }
}
