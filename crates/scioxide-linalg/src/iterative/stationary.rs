//! Richardson, Jacobi, Gauss–Seidel and SOR iterations.
use super::biorthogonal::{Actions, Config, Kernel, Operator};
use super::safety::Step;
use super::support::validate_square;
use super::workspace::{IterativeProgress, Job, Shape, StorageKind};
use crate::Preconditioning;
use crate::kernel::finite;
use crate::{
    FiniteMatrixView, IterativeLinearSolver, IterativeRequirements, IterativeSolveReport,
    IterativeSolveResult, IterativeWorkspace, IterativeWorkspaceRequest, LinalgExecutionContext,
    LinearSolveTermination, Vector, VectorView, VectorViewMut,
};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, NumericScalar, RealScalar, ResidualTolerance,
    SciOxideError,
};
use scioxide_sparse::{CscMatrix, CsrMatrix};
type Result<T> = core::result::Result<T, SciOxideError>;

/// Relaxation weight `ω > 0` of a Richardson step or a weighted Jacobi sweep.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::RelaxationWeight;
/// assert_eq!(RelaxationWeight::new(0.5_f64)?.get(), 0.5);
/// assert!(RelaxationWeight::new(0.0_f64).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RelaxationWeight<R: RealScalar>(R);

impl<R: RealScalar> RelaxationWeight<R> {
    /// Creates the weight `value`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `value ≤ 0`.
    pub fn new(value: R) -> Result<Self> {
        weight(value, R::INFINITY, "relaxation weight must be positive").map(Self)
    }

    /// Returns the weight `ω = 1` of unweighted Jacobi and plain Richardson.
    pub const fn one() -> Self {
        Self(R::ONE)
    }

    /// Returns `ω`.
    pub const fn get(self) -> R {
        self.0
    }
}

/// Successive over-relaxation weight `ω` in the open interval `(0, 2)`.
///
/// By Kahan's theorem, SOR can converge only for `0 < ω < 2`, so other values
/// are rejected when the weight is created.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::SorWeight;
/// assert_eq!(SorWeight::new(1.5_f64)?.get(), 1.5);
/// assert!(SorWeight::new(2.0_f64).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SorWeight<R: RealScalar>(R);

impl<R: RealScalar> SorWeight<R> {
    /// Creates the weight `value`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `value` is outside `(0, 2)`.
    pub fn new(value: R) -> Result<Self> {
        weight(
            value,
            R::TWO,
            "SOR weight must lie in the open interval (0, 2)",
        )
        .map(Self)
    }

    /// Returns `ω`.
    pub const fn get(self) -> R {
        self.0
    }
}

/// A finite weight in the open interval `(0, upper)`.
fn weight<R: RealScalar>(value: R, upper: R, detail: &'static str) -> Result<R> {
    let value = crate::checked::finite(
        value,
        scioxide_core::NonFiniteOrigin::Input,
        "weight must be finite",
    )?;
    if value > R::ZERO && value < upper {
        Ok(value)
    } else {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::OutOfRange,
            detail,
        ))
    }
}

#[derive(Debug)]
pub(super) struct Richardson<S: NumericScalar> {
    r: Vector<S>,
    temp: Vector<S>,
}
impl<S: NumericScalar> Kernel<S> for Richardson<S> {
    const TRUE_RESIDUAL_EACH_STEP: bool = true;
    fn residual_norm(&self, c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real> {
        c.execution().check()?;
        crate::kernel::standard::norm(self.r.as_slice().iter().copied())
    }
    const KIND: StorageKind = StorageKind::Richardson;
    const VECTORS: usize = 2;
    const KRYLOV_DIAGNOSTICS: bool = false;
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            r: Vector::zeros(n)?,
            temp: Vector::zeros(n)?,
        })
    }
    fn start(&mut self, r: &Vector<S>) -> Result<()> {
        finite::copy(self.r.as_mut_slice(), r.as_slice())
    }
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        // The original residual is refreshed by the driver after every step.
        Ok(a.update(x, &self.r, S::from_real(a.relaxation), &mut self.temp, c)?)
    }
}
/// Preconditioned Richardson iteration `xₖ₊₁ = xₖ + ω M⁻¹ (b − A xₖ)` for square systems.
///
/// The preconditioner `M⁻¹` is applied from the left and must not change during
/// a solve. The iteration converges for every `x₀` exactly when the spectral
/// radius of `I − ω M⁻¹ A` is below one. Each iteration costs one product with
/// `A` and one application of `M⁻¹`. The residual `b − A xₖ` is recomputed in
/// every iteration, so the convergence test `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`
/// always uses the true residual. The workspace holds eight vectors of length `n`.
#[derive(Clone)]
#[doc = r#"

# Examples

```
use scioxide_core::{Budget, ResidualTolerance};
use scioxide_linalg::*;
let a = Matrix::<f64>::identity(2)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let tolerance = ResidualTolerance::new(1e-10, 0.0)?;
let budget = Budget::try_from(20_usize)?;
let solver = RichardsonSolver::new(tolerance, budget, RelaxationWeight::new(1.0)?);
let result = solver.solve(&a.view().try_finite()?, &rhs)?;
assert!((result.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
"#]
pub struct RichardsonSolver<S: NumericScalar> {
    config: Config<S>,
}
impl<S: NumericScalar> core::fmt::Debug for RichardsonSolver<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RichardsonSolver")
            .field("max_iterations", &self.config.max_iterations)
            .field("execution", &self.config.execution)
            .field("relaxation", &self.config.relaxation)
            .field("history", &self.config.history)
            .finish_non_exhaustive()
    }
}
impl<S: NumericScalar> RichardsonSolver<S> {
    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.config.execution
    }

    /// Sets the execution options: policy, product accuracy, time limit, byte
    /// limit and iteration budget.
    ///
    /// An iteration budget in `execution` caps `limit`. The accuracy setting
    /// applies to dense operator products; the recurrence safeguards always use
    /// `Standard` accuracy.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.config.execution = execution;
        self
    }

    /// Creates a solver with residual tolerance `tolerance`, at most `limit`
    /// iterations and relaxation weight `relaxation`.
    pub fn new(
        tolerance: ResidualTolerance<S::Real>,
        limit: Budget,
        relaxation: RelaxationWeight<S::Real>,
    ) -> Self {
        let mut config = Config::new(tolerance, limit);
        config.relaxation = relaxation.get();
        Self { config }
    }

    /// Enables or disables the history of true residual norms.
    #[must_use]
    pub fn with_history(mut self, enabled: bool) -> Self {
        self.config.history = enabled;
        self
    }
    /// Returns the storage a solve of `request` needs, including preconditioner scratch.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the execution options.
    pub fn workspace_requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>> {
        self.config.requirements(StorageKind::Richardson, request)
    }
    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn workspace(&self, request: &IterativeWorkspaceRequest) -> Result<IterativeWorkspace<S>> {
        let mut workspace = IterativeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
    ///
    /// A solve that ends without meeting the tolerance, at the iteration or
    /// evaluation limit or by a breakdown, is an error. Use
    /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
    /// start vector or the iterate of an unsuccessful solve.
    ///
    /// # Errors
    ///
    /// - The errors of [`solve_with_options`](Self::solve_with_options).
    /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
    ///   diagnostics name the solver and carry the iteration count and the
    ///   residual norm.
    pub fn solve<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
    ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
        self.solve_with_options(a, b, crate::Preconditioning::None, None)?
            .into_converged_solution()
    }
    /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is `None`, and returns
    /// the iterate with its report.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn solve_with_options<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        x0: Option<VectorView<'_, S>>,
    ) -> Result<IterativeSolveResult<S>> {
        let b = b.into();
        IterativeWorkspace::solve_owned(a.cols(), x0, self.algorithm_id(), |x, workspace| {
            self.solve_into(
                a,
                b,
                preconditioning,
                x,
                workspace,
                ExecutionContext::sequential_native(),
            )
        })
    }
    /// Solves `A x = b` in place, starting from the values in `x`.
    ///
    /// # Errors
    ///
    /// The errors listed on
    /// [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        x: VectorViewMut<'_, S>,
        workspace: &mut IterativeWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<IterativeSolveReport<S::Real>> {
        super::biorthogonal::solve_into::<S, Richardson<S>>(
            &self.config,
            self.algorithm_id(),
            Operator::Forward(a),
            b.into(),
            preconditioning,
            x,
            workspace,
            context.into(),
        )
    }
}
crate::traits::impl_iterative_solver!(
    RichardsonSolver,
    _plan,
    scioxide_core::algorithm_id!("linear", "Richardson", 1)
);

/// Matrix whose entries the Jacobi, Gauss–Seidel and SOR solvers read.
///
/// The splitting methods need the diagonal and the strictly lower triangle of
/// `A`, so they take matrix entries rather than a [`LinearOperator`]. A dense
/// [`Matrix`](crate::Matrix) enters through `view().try_finite()`.
#[derive(Clone, Copy)]
pub enum StationaryMatrix<'a, S: NumericScalar> {
    /// A dense matrix, including strided or conjugated views.
    Dense(FiniteMatrixView<'a, S>),
    /// A compressed sparse row (CSR) matrix.
    Csr(&'a CsrMatrix<S>),
    /// A compressed sparse column (CSC) matrix.
    Csc(&'a CscMatrix<S>),
}
impl<S: NumericScalar> StationaryMatrix<'_, S> {
    pub(super) fn operator(&self) -> &dyn LinearOperator<S> {
        match self {
            Self::Dense(a) => a,
            Self::Csr(a) => *a,
            Self::Csc(a) => *a,
        }
    }
}
/// Storage for diagonal or successive row corrections, shared by the stationary methods.
#[derive(Debug)]
pub(super) struct StationaryStorage<S: NumericScalar> {
    x: Vector<S>,
    accepted: Vector<S>,
    best: super::support::BestIterate<S>,
    residual: Vector<S>,
    product: Vector<S>,
    correction: Vector<S>,
    diagonal: Vec<S>,
    scratch: Vec<S>,
    history: Option<Vec<S::Real>>,
}
#[derive(Clone, Copy)]
enum Method {
    Jacobi,
    GaussSeidel,
    Sor,
}
impl Method {
    fn algorithm_id(self) -> scioxide_core::AlgorithmId {
        match self {
            Self::Jacobi => scioxide_core::algorithm_id!("linear", "Jacobi", 1),
            Self::GaussSeidel => scioxide_core::algorithm_id!("linear", "Gauss-Seidel", 1),
            Self::Sor => scioxide_core::algorithm_id!("linear", "SOR", 1),
        }
    }
}
#[derive(Clone)]
struct Engine<S: NumericScalar> {
    config: Config<S>,
    method: Method,
}
impl<S: NumericScalar> Engine<S> {
    fn requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>> {
        IterativeRequirements::new(
            Shape {
                kind: StorageKind::Stationary,
                rows: request.rows(),
                cols: request.cols(),
                operator_scratch: request.operator_scratch(),
                preconditioner_scratch: 0,
                columns: 1,
                cycle: 0,
                retained: 0,
            },
            self.config.history.then_some(self.config.limit()),
            Some(request),
            self.config.execution.byte_limit(),
        )
    }
    fn solve(
        &self,
        a: StationaryMatrix<'_, S>,
        b: &Vector<S>,
        x: &mut Vector<S>,
        w: &mut StationaryStorage<S>,
        kernels: crate::KernelContext<'_>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveReport<S::Real>> {
        let op = a.operator();
        w.prepare(a)?;
        let reference = finite::checked_norm(b.as_slice(), c.execution())?;
        let scale = super::residual_reference(reference);

        finite::copy(w.x.as_mut_slice(), x.as_slice())?;
        finite::copy(w.accepted.as_mut_slice(), x.as_slice())?;
        if let Some(h) = &mut w.history {
            h.clear();
        }
        let Some(mut verified) = super::support::true_residual_into_scoped(
            super::support::KrylovProblem {
                operator: op,
                rhs: b,
            },
            &w.x,
            &mut w.residual,
            &mut w.product,
            &mut w.scratch,
            kernels,
            c,
        )?
        else {
            return Err(super::evaluation_limit_error(c));
        };
        w.best.reset(w.x.as_slice(), verified);
        let floor = super::safety::ResidualFloor::new(reference);
        c.assess_accepted(super::checkpoint_trust(
            self.config.tolerance.is_satisfied(verified, scale)?,
        ))?;
        if self.config.history
            && let Some(h) = &mut w.history
        {
            h.push(verified);
        }
        let outcome = (|| -> Result<LinearSolveTermination> {
            loop {
                if self.config.tolerance.is_satisfied(verified, scale)? {
                    return Ok(LinearSolveTermination::Converged);
                }
                if let Some(reason) = c.stop_reason()? {
                    return Ok(LinearSolveTermination::ExecutionStopped(reason));
                }
                if c.counts().own().iterations() >= super::support::iteration_limit(c)
                    || !c.record_iteration()?.is_granted()
                {
                    return Ok(LinearSolveTermination::IterationLimit);
                }
                w.sweep(a, self.method, self.config.relaxation, kernels)?;
                let Some(next) = super::support::true_residual_into_scoped(
                    super::support::KrylovProblem {
                        operator: op,
                        rhs: b,
                    },
                    &w.x,
                    &mut w.residual,
                    &mut w.product,
                    &mut w.scratch,
                    kernels,
                    c,
                )?
                else {
                    return Ok(LinearSolveTermination::EvaluationLimit);
                };
                let unchanged = w.x.as_slice() == w.accepted.as_slice();
                let best_before = w.best.residual;
                let iteration = c.counts().own().iterations();
                w.best.observe(w.x.as_slice(), next, iteration);
                finite::copy(w.accepted.as_mut_slice(), w.x.as_slice())?;
                verified = next;
                let satisfied = self.config.tolerance.is_satisfied(verified, scale)?;
                c.commit_accepted(super::checkpoint_trust(satisfied))?;
                if self.config.history
                    && let Some(h) = &mut w.history
                {
                    h.push(verified);
                }
                if satisfied {
                    return Ok(LinearSolveTermination::Converged);
                }
                if unchanged {
                    return Ok(LinearSolveTermination::Stagnation);
                }
                // Every step is verified, so only the periodic checkpoints pay
                // for the floor probe; the correction is recomputed by the next sweep.
                if iteration < super::support::iteration_limit(c)
                    && super::safety::should_recompute_true_residual(iteration)
                    && super::safety::floor_test_due(verified, best_before)
                {
                    match floor.reached(
                        op,
                        w.best.solution.as_slice(),
                        w.best.residual,
                        super::safety::FloorProbe {
                            input: w.correction.as_mut_slice(),
                            image: w.product.as_mut_slice(),
                            workspace: &mut w.scratch,
                        },
                        kernels,
                        c,
                    )? {
                        None => return Ok(LinearSolveTermination::EvaluationLimit),
                        Some(true) => return Ok(LinearSolveTermination::Stagnation),
                        Some(false) => {}
                    }
                }
            }
        })();
        let termination = match outcome {
            Ok(t) => t,
            Err(error) => match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. } => {
                    super::support::execution_stop(error)?
                }
                _ => return Err(error),
            },
        };
        if !termination.is_success() {
            w.accepted
                .as_mut_slice()
                .copy_from_slice(w.best.solution.as_slice());
            verified = w.best.residual;
        }
        if let Some(last) = w.history.as_mut().and_then(|h| h.last_mut()) {
            *last = verified;
        }
        let report = IterativeSolveReport::from_summary(super::solve_summary(
            c,
            termination,
            super::TrueResidual::against(verified, scale),
        ))?;
        finite::copy(x.as_mut_slice(), w.accepted.as_slice())?;
        Ok(report)
    }
}
impl<S: NumericScalar> StationaryStorage<S> {
    /// Exact payload of [`Self::new`]: seven vectors, provider scratch, and history.
    pub(super) fn bytes(shape: &Shape, history: usize) -> Result<usize> {
        use super::workspace::{times, total};
        let n = shape.cols;
        super::workspace::payload::<S>(total([times(7, n)?, shape.operator_scratch])?, history)
    }

    pub(super) fn new(shape: &Shape, history: usize) -> Result<Self> {
        let n = shape.cols;
        Ok(Self {
            x: Vector::zeros(n)?,
            accepted: Vector::zeros(n)?,
            best: super::support::BestIterate::new(n)?,
            residual: Vector::zeros(n)?,
            product: Vector::zeros(n)?,
            correction: Vector::zeros(n)?,
            diagonal: try_filled(n, S::ZERO, scioxide_core::quantity!("workspace.diagonal"))?,
            scratch: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("workspace.scratch"),
            )?,
            history: (history > 0)
                .then(|| try_with_capacity(history, scioxide_core::quantity!("workspace.history")))
                .transpose()?,
        })
    }

    pub(super) fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }

    fn prepare(&mut self, a: StationaryMatrix<'_, S>) -> Result<()> {
        self.diagonal.fill(S::ZERO);
        for row in 0..self.x.len() {
            let mut diagonal = S::ZERO;
            self.row(a, row, |col, value| {
                crate::checked::validate_input(value, "matrix entry")?;
                if col == row {
                    diagonal = value;
                }
                Ok(())
            })?;
            if diagonal == S::ZERO {
                return Err(scioxide_core::SciOxideError::singular(None));
            }
            self.diagonal[row] = diagonal;
        }
        Ok(())
    }
    fn row(
        &self,
        a: StationaryMatrix<'_, S>,
        row: usize,
        mut visit: impl FnMut(usize, S) -> Result<()>,
    ) -> Result<()> {
        match a {
            StationaryMatrix::Dense(m) => {
                (0..m.cols()).try_for_each(|col| visit(col, m.get(row, col)?))
            }
            StationaryMatrix::Csr(m) => (m.row_ptr()[row]..m.row_ptr()[row + 1])
                .try_for_each(|i| visit(m.col_indices()[i], m.values()[i])),
            StationaryMatrix::Csc(m) => {
                let traversal = m.opposite_traversal();
                let pattern = traversal.pattern();
                (pattern.row_ptr()[row]..pattern.row_ptr()[row + 1]).try_for_each(|k| {
                    visit(
                        pattern.col_indices()[k],
                        m.values()[traversal.positions()[k]],
                    )
                })
            }
        }
    }
    fn sweep(
        &mut self,
        a: StationaryMatrix<'_, S>,
        method: Method,
        omega: S::Real,
        kernels: crate::KernelContext<'_>,
    ) -> Result<()> {
        // Residual-correction form of the splitting: `x += M^{-1} r` with
        // `M = D / omega` (Jacobi) or `M = D / omega + L` (Gauss-Seidel for
        // omega = 1, SOR otherwise). The latter is a forward substitution:
        // `dx_i = omega (r_i - sum_{j<i} a_ij dx_j) / a_ii`, algebraically
        // identical to the classical in-place sweep, but driven by the
        // verified true residual of the accepted iterate.
        self.correction.view_mut().fill(S::ZERO);
        for row in 0..self.x.len() {
            let mut residual = self.residual.as_slice()[row];
            if !matches!(method, Method::Jacobi) {
                self.row(a, row, |col, value| {
                    if col < row {
                        residual = crate::checked::sub(
                            residual,
                            crate::checked::mul(value, self.correction.as_slice()[col])?,
                        )?;
                    }
                    Ok(())
                })?;
            }
            let correction = crate::checked::div(residual, self.diagonal[row])?;
            self.correction.as_mut_slice()[row] =
                crate::checked::mul(S::from_real(omega), correction)?;
        }
        finite::copy(self.x.as_mut_slice(), self.accepted.as_slice())?;
        finite::update(
            self.x.as_mut_slice(),
            S::ONE,
            self.correction.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )
        .map(|_| ())
    }
}

macro_rules! matrix_solver {
    ($solver:ident, $example:literal,$method:ident) => {
        #[doc=concat!(stringify!($method)," stationary iteration with verified residual stopping.")]
        #[derive(Clone)]
        #[doc = $example]
        pub struct $solver<S: NumericScalar> {
            engine: Engine<S>,
        }
        impl<S: NumericScalar> core::fmt::Debug for $solver<S> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($solver))
                    .field("max_iterations", &self.engine.config.max_iterations)
                    .field("execution", &self.engine.config.execution)
                    .field("relaxation", &self.engine.config.relaxation)
                    .field("history", &self.engine.config.history)
                    .finish()
            }
        }
        impl<S: NumericScalar> $solver<S> {
            /// Returns the execution options.
            pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
                self.engine.config.execution
            }

            /// Sets the execution options: policy, product accuracy, time limit,
            /// byte limit and iteration budget.
            ///
            /// An iteration budget in `execution` caps `limit`. The accuracy
            /// setting applies to dense operator products; the recurrence
            /// safeguards always use `Standard` accuracy.
            pub const fn with_execution_options(
                mut self,
                execution: crate::LinalgExecutionOptions,
            ) -> Self {
                self.engine.config.execution = execution;
                self
            }

            /// Enables or disables the history of true residual norms.
            ///
            /// The history holds the initial residual and the residual after each
            /// iteration.
            #[must_use]
            pub fn with_history(mut self, enabled: bool) -> Self {
                self.engine.config.history = enabled;
                self
            }
            /// Returns the storage a solve of `request` needs.
            ///
            /// Build `request` with [`IterativeWorkspaceRequest::matrix`].
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if a size or byte count overflows `usize`.
            /// - `ResourceLimit` if the storage exceeds the byte limit of the
            ///   execution options.
            pub fn workspace_requirements(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeRequirements<S>> {
                self.engine.requirements(request)
            }
            /// Creates a workspace reserved for `request`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::workspace_requirements`].
            /// - `AllocationFailure` if memory cannot be reserved.
            pub fn workspace(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeWorkspace<S>> {
                let mut workspace = IterativeWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(request)?)?;
                Ok(workspace)
            }
            /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
            ///
            /// A solve that ends without meeting the tolerance, at the iteration or
            /// evaluation limit or by a breakdown, is an error. Use
            /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
            /// start vector or the iterate of an unsuccessful solve.
            ///
            /// # Errors
            ///
            /// - The errors of [`solve_with_options`](Self::solve_with_options).
            /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
            ///   diagnostics name the solver and carry the iteration count and the
            ///   residual norm.
            pub fn solve<'b>(
                &self,
                a: StationaryMatrix<'_, S>,
                b: impl Into<VectorView<'b, S>>,
            ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
                self.solve_with_options(a, b, None)?
                    .into_converged_solution()
            }
            /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is `None`, and
            /// returns the iterate with its report.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `AllocationFailure` if the output or workspace cannot be allocated.
            pub fn solve_with_options<'b>(
                &self,
                a: StationaryMatrix<'_, S>,
                b: impl Into<VectorView<'b, S>>,
                x0: Option<VectorView<'_, S>>,
            ) -> Result<IterativeSolveResult<S>> {
                let b = b.into();
                IterativeWorkspace::solve_owned(
                    a.operator().cols(),
                    x0,
                    self.engine.method.algorithm_id(),
                    |x, workspace| {
                        self.solve_into(a, b, x, workspace, ExecutionContext::sequential_native())
                    },
                )
            }
            /// Solves `A x = b` in place, starting from the values in `x`.
            ///
            /// A sweep is internal work: only the products with `A` for the
            /// residual and the rounding-floor test count toward the evaluation
            /// limit.
            ///
            /// # Errors
            ///
            /// - The errors listed on
            ///   [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            /// - `Singular` if a diagonal entry of `A` is exactly zero.
            /// - `NonFinite` if an entry of `A` is NaN or infinite.
            pub fn solve_into<'b, 'scope, 'capability>(
                &self,
                a: StationaryMatrix<'_, S>,
                b: impl Into<VectorView<'b, S>>,
                mut x: VectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<IterativeSolveReport<S::Real>> {
                workspace.reject();
                validate_square(a.operator())?;
                let requirements =
                    self.workspace_requirements(&IterativeWorkspaceRequest::matrix(a)?)?;
                workspace.run(
                    context.into().with_options(&self.engine.config.execution)?,
                    &Job {
                        algorithm: self.engine.method.algorithm_id(),
                        limit: self.engine.config.limit(),
                        requirements,
                    },
                    b.into(),
                    &mut x,
                    |storage, b, x, kernels, solve| {
                        self.engine.solve(a, b, x, storage, kernels, solve)
                    },
                    IterativeProgress::of,
                )
            }
        }
    };
}
matrix_solver!(
    JacobiSolver,
    r#"

The Jacobi iteration is `xₖ₊₁ = xₖ + ω D⁻¹ (b − A xₖ)`, where `D` is the
diagonal of `A` and `ω` is the weight set with [`JacobiSolver::with_weight`]
(default 1). Every diagonal entry must be nonzero. The iteration converges for
every `x₀` when `A` is strictly diagonally dominant and `0 < ω ≤ 1`, and in
general exactly when the spectral radius of `I − ω D⁻¹ A` is below one. Each iteration costs `O(n)`
for the correction and one product with `A` for the true residual, which the
convergence test `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂` uses. The workspace holds seven
vectors of length `n`.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, JacobiSolver, LinearSolveTermination, Matrix,
    Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let matrix = scioxide_linalg::StationaryMatrix::Dense(operator);
let solver = JacobiSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::matrix(matrix)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    matrix,
    &rhs,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    Jacobi
);
matrix_solver!(
    GaussSeidelSolver,
    r#"

The Gauss–Seidel iteration is `xₖ₊₁ = xₖ + (D + L)⁻¹ (b − A xₖ)`, where `D` is
the diagonal and `L` the strictly lower triangle of `A`; rows are processed in
natural order. Every diagonal entry must be nonzero. The iteration converges
for every `x₀` when `A` is strictly diagonally dominant or Hermitian positive
definite. Each iteration costs one forward substitution with `D + L` and one
product with `A` for the true residual, which the convergence test
`‖b − A x‖₂ ≤ atol + rtol·‖b‖₂` uses. It equals [`SorSolver`] with `ω = 1`.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    GaussSeidelSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination,
    Matrix, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let matrix = scioxide_linalg::StationaryMatrix::Dense(operator);
let solver = GaussSeidelSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::matrix(matrix)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    matrix,
    &rhs,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    GaussSeidel
);
matrix_solver!(
    SorSolver,
    r#"

Successive over-relaxation (SOR) is `xₖ₊₁ = xₖ + (D/ω + L)⁻¹ (b − A xₖ)`, where
`D` is the diagonal and `L` the strictly lower triangle of `A`, and
`0 < ω < 2`. Every diagonal entry must be nonzero. For Hermitian positive
definite `A`, SOR converges for every `ω` in `(0, 2)`. Each iteration costs one
forward substitution with `D/ω + L` and one product with `A` for the true
residual, which the convergence test `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂` uses.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix, SorSolver,
    Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let matrix = scioxide_linalg::StationaryMatrix::Dense(operator);
let solver = SorSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
    scioxide_linalg::SorWeight::new(1.0)?,
);
let request = IterativeWorkspaceRequest::matrix(matrix)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    matrix,
    &rhs,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    Sor
);
impl<S: NumericScalar> JacobiSolver<S> {
    /// Creates a Jacobi solver with weight one, residual tolerance `tolerance` and
    /// at most `limit` iterations.
    pub fn new(tolerance: ResidualTolerance<S::Real>, limit: Budget) -> Self {
        Self {
            engine: Engine {
                config: Config::new(tolerance, limit),
                method: Method::Jacobi,
            },
        }
    }
    /// Sets the weight `ω` of weighted Jacobi.
    ///
    /// Convergence depends on the spectrum of `D⁻¹ A`.
    #[must_use]
    pub fn with_weight(mut self, omega: RelaxationWeight<S::Real>) -> Self {
        self.engine.config.relaxation = omega.get();
        self
    }
}
impl<S: NumericScalar> GaussSeidelSolver<S> {
    /// Creates a Gauss–Seidel solver with residual tolerance `tolerance` and at
    /// most `limit` iterations.
    pub fn new(tolerance: ResidualTolerance<S::Real>, limit: Budget) -> Self {
        Self {
            engine: Engine {
                config: Config::new(tolerance, limit),
                method: Method::GaussSeidel,
            },
        }
    }
}
impl<S: NumericScalar> SorSolver<S> {
    /// Creates an SOR solver with residual tolerance `tolerance`, at most `limit`
    /// iterations and weight `omega`.
    pub fn new(
        tolerance: ResidualTolerance<S::Real>,
        limit: Budget,
        omega: SorWeight<S::Real>,
    ) -> Self {
        let mut config = Config::new(tolerance, limit);
        config.relaxation = omega.get();
        Self {
            engine: Engine {
                config,
                method: Method::Sor,
            },
        }
    }
}
