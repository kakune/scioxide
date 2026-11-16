//! MINRES (Lanczos) and LSQR and LSMR (Golub–Kahan bidiagonalization).
//!
//! None of the three methods forms the normal equations.

use super::safety::{Step, StepFault};
use super::workspace::{IterativeProgress, Job, Shape, StorageKind};
use super::{TrueResidual, evaluation_limit_error, solve_summary};
use crate::Preconditioning;
use crate::checked::dimension_epsilon;
use crate::kernel::finite;
use crate::{
    ConditionEstimate, IterativeRequirements, IterativeSolveReport, IterativeSolveResult,
    IterativeWorkspace, IterativeWorkspaceRequest, KrylovDiagnostics, LinalgExecutionContext,
    LinearSolveTermination, ReciprocalConditionThreshold, Vector, VectorView, VectorViewMut,
};
use alloc::vec::Vec;
use core::fmt;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    AdjointOperator, Budget, EstimateReliability, ExecutionContext, GradientTolerance,
    LinearOperator, NormSelection, NumericScalar, RealScalar, ResidualReport, ResidualScaling,
    ResidualScalingRule, ResidualTolerance, SciOxideError,
};

type Result<T> = core::result::Result<T, SciOxideError>;

struct VerifiedResiduals<R> {
    residual: R,
    normal: Option<R>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Minres,
    Lsqr,
    Lsmr,
}
impl Method {
    const fn algorithm_id(self) -> scioxide_core::AlgorithmId {
        match self {
            Self::Minres => scioxide_core::algorithm_id!("linear", "MINRES", 1),
            Self::Lsqr => scioxide_core::algorithm_id!("linear", "LSQR", 1),
            Self::Lsmr => scioxide_core::algorithm_id!("linear", "LSMR", 1),
        }
    }
    const fn name(self) -> &'static str {
        match self {
            Self::Minres => "MINRES",
            Self::Lsqr => "LSQR",
            Self::Lsmr => "LSMR",
        }
    }
}

#[derive(Clone, Copy)]
enum Operator<'a, S: NumericScalar> {
    Hermitian(&'a dyn LinearOperator<S>),
    Rectangular(&'a dyn AdjointOperator<S>),
}
impl<'a, S: NumericScalar> Operator<'a, S> {
    fn linear(self) -> &'a dyn LinearOperator<S> {
        match self {
            Self::Hermitian(a) => a,
            Self::Rectangular(a) => a,
        }
    }
    fn adjoint_workspace_len(self) -> Result<usize> {
        match self {
            Self::Hermitian(a) => a.apply_workspace_len(),
            Self::Rectangular(a) => a.apply_adjoint_workspace_len(),
        }
    }
    fn apply(
        self,
        adjoint: bool,
        input: &Vector<S>,
        output: &mut Vector<S>,
        scratch: &mut [S],
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool> {
        let len = if adjoint {
            self.adjoint_workspace_len()?
        } else {
            self.linear().apply_workspace_len()?
        };
        let bound = finite::maximum(input.as_slice(), kernels.dispatch())?;
        let invocation = match (self, adjoint) {
            (Self::Rectangular(a), true) => super::safety::apply_adjoint_scoped(
                a,
                input.as_slice(),
                bound,
                output.as_mut_slice(),
                &mut scratch[..len],
                context,
            )?,
            _ => super::safety::apply_operator_scoped(
                self.linear(),
                input.as_slice(),
                bound,
                output.as_mut_slice(),
                &mut scratch[..len],
                context,
            )?,
        };
        Ok(!invocation.is_exhausted())
    }
}

struct KrylovProgress<'a, R: RealScalar> {
    verified: &'a mut (R, R),
    diagnostics: &'a mut KrylovDiagnostics<R>,
    /// Convergence reference, see [`super::residual_reference`].
    scale: R,
    /// Raw right-hand side norm for the rounding model.
    rhs_norm: R,
}
#[derive(Clone)]
struct Engine<S: NumericScalar> {
    method: Method,
    tolerance: ResidualTolerance<S::Real>,
    normal_tolerance: Option<GradientTolerance<S::Real>>,
    max_iterations: Budget,
    execution: crate::LinalgExecutionOptions,
    condition_limit: S::Real,
    history: bool,
}
impl<S: NumericScalar> fmt::Debug for Engine<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(self.method.name())
            .field("tolerance", &self.tolerance)
            .field("normal_tolerance", &self.normal_tolerance)
            .field("max_iterations", &self.max_iterations)
            .field("execution", &self.execution)
            .field("condition_limit", &self.condition_limit)
            .field("history", &self.history)
            .finish()
    }
}

#[derive(Debug)]
pub(super) struct Storage<S: NumericScalar> {
    rows: usize,
    cols: usize,
    x: Vector<S>,
    accepted: Vector<S>,
    best: super::support::BestIterate<S>,
    best_normal: Option<S::Real>,
    residual: Vector<S>,
    normal: Vector<S>,
    u: Vector<S>,
    v: Vector<S>,
    previous_u: Vector<S>,
    next_v: Vector<S>,
    product: Vector<S>,
    w: Vector<S>,
    previous_w: Vector<S>,
    hbar: Vector<S>,
    scratch: Vec<S>,
    preconditioner_scratch: Vec<S>,
    minres_images: [Vector<S>; 3],
    minres_residual: Vector<S>,
    history: Option<Vec<S::Real>>,
}
impl<S: NumericScalar> Storage<S> {
    /// Exact payload of [`Self::new`]: eight row and nine column vectors,
    /// provider scratch and history.
    pub(super) fn bytes(shape: &Shape, history: usize) -> Result<usize> {
        use super::workspace::{times, total};
        super::workspace::payload::<S>(
            total([
                times(8, shape.rows)?,
                times(9, shape.cols)?,
                shape.operator_scratch,
                shape.preconditioner_scratch,
            ])?,
            history,
        )
    }

    pub(super) fn new(shape: &Shape, history: usize) -> Result<Self> {
        let (rows, cols) = (shape.rows, shape.cols);
        Ok(Self {
            rows,
            cols,
            minres_images: [
                Vector::zeros(rows)?,
                Vector::zeros(rows)?,
                Vector::zeros(rows)?,
            ],
            minres_residual: Vector::zeros(rows)?,
            x: Vector::zeros(cols)?,
            accepted: Vector::zeros(cols)?,
            best: super::support::BestIterate::new(cols)?,
            best_normal: None,
            residual: Vector::zeros(rows)?,
            normal: Vector::zeros(cols)?,
            u: Vector::zeros(rows)?,
            v: Vector::zeros(cols)?,
            previous_u: Vector::zeros(rows)?,
            next_v: Vector::zeros(cols)?,
            product: Vector::zeros(rows)?,
            w: Vector::zeros(cols)?,
            previous_w: Vector::zeros(cols)?,
            hbar: Vector::zeros(cols)?,
            scratch: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("krylov_operator_scratch"),
            )?,
            preconditioner_scratch: try_filled(
                shape.preconditioner_scratch,
                S::ZERO,
                scioxide_core::quantity!("krylov_preconditioner_scratch"),
            )?,
            history: (history > 0)
                .then(|| try_with_capacity(history, scioxide_core::quantity!("krylov_history")))
                .transpose()?,
        })
    }

    pub(super) fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }
}

// The public wrappers preserve each method's required operator capability.
// Only API plumbing is generated; there is one numerical implementation per method.
macro_rules! solver {
    (@inverse) => { Preconditioning::None };
    (@inverse $value:ident) => { $value };
    (@none $value:ident) => { Preconditioning::None };
    ($solver:ident, $example:literal, $method:ident, $operator:ident, $variant:ident, $request:ident, $description:literal $(, $preconditioning:ident)?) => {
        #[doc = $description]
        ///
        /// Works for real and complex scalars in `f32` and `f64`.
        #[derive(Debug, Clone)]
        #[doc = $example]
        pub struct $solver<S: NumericScalar> {
            engine: Engine<S>,
        }
        impl<S: NumericScalar> $solver<S> {
            /// Returns the execution options.
            pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
                self.engine.execution
            }

            /// Sets the execution options: policy, product accuracy, time limit,
            /// byte limit and iteration budget.
            ///
            /// An iteration budget in `execution` caps `max_iterations`. The
            /// accuracy setting applies to dense operator products; the
            /// recurrence safeguards always use `Standard` accuracy.
            pub const fn with_execution_options(
                mut self,
                execution: crate::LinalgExecutionOptions,
            ) -> Self {
                self.engine.execution = execution;
                self
            }

            /// Creates a solver with residual tolerance `tolerance` and at most
            /// `max_iterations` iterations.
            ///
            /// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`, with
            /// `‖b‖₂` replaced by one when `b = 0`. Until
            /// [`Self::with_normal_tolerance`] sets it, the normal-residual test
            /// uses absolute tolerance zero and the relative tolerance `rtol`.
            /// The condition limit is off and no history is recorded.
            #[must_use]
            pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
                Self {
                    engine: Engine {
                        method: Method::$method,
                        tolerance,
                        normal_tolerance: None,
                        max_iterations,
                        execution: crate::LinalgExecutionOptions::new(),
                        condition_limit: S::Real::ZERO,
                        history: false,
                    },
                }
            }
            /// Sets the tolerance of the normal-residual test.
            ///
            /// Without preconditioning, the solve stops with
            /// [`LeastSquaresConverged`](crate::LinearSolveTermination::LeastSquaresConverged)
            /// when the true residual `r = b − A x` satisfies
            /// `‖Aᴴ r‖₂ ≤ atol + rtol·‖A‖·‖r‖₂`. `‖A‖` is the norm estimate built
            /// by the recurrence, and `atol` has the units of `‖Aᴴ r‖₂`. This test
            /// detects least-squares solutions of inconsistent or singular systems,
            /// whose residual cannot reach the residual tolerance.
            #[must_use]
            pub fn with_normal_tolerance(mut self, tolerance: GradientTolerance<S::Real>) -> Self {
                self.engine.normal_tolerance = Some(tolerance);
                self
            }
            /// Stops with [`ConditionLimit`](crate::LinearSolveTermination::ConditionLimit)
            /// when the reciprocal condition estimate falls below `threshold`.
            ///
            /// The estimate comes from the projected problem of the current
            /// recurrence. It is a heuristic, not a bound on the condition number
            /// of `A`. A threshold of zero disables this stop.
            #[must_use]
            pub fn with_condition_limit(
                mut self,
                threshold: ReciprocalConditionThreshold<S::Real>,
            ) -> Self {
                self.engine.condition_limit = threshold.get();
                self
            }
            /// Enables or disables the residual history.
            ///
            /// Between verification checkpoints the history holds the residual
            /// estimates of the recurrence. The initial and final samples are true
            /// residual norms.
            #[must_use]
            pub fn with_history(mut self, enabled: bool) -> Self {
                self.engine.history = enabled;
                self
            }
            /// Returns the storage a solve of `request` needs.
            ///
            /// The storage is 8 vectors of length `request.rows()`, 9 of length
            /// `request.cols()`, the operator and preconditioner scratch, and
            /// `max_iterations + 1` reals when history is enabled.
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
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
            ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
                self.solve_with_options(a, b, $( solver!(@none $preconditioning), )? None)?.into_converged_solution()
            }
            /// Solves from the initial guess `x0`, or from zero when `x0` is `None`, and returns
            /// the iterate with its report.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `AllocationFailure` if the output or workspace cannot be allocated.
            pub fn solve_with_options<'b>(
                &self,
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
                $( $preconditioning: Preconditioning<'_, S>, )?
                x0: Option<VectorView<'_, S>>,
            ) -> Result<IterativeSolveResult<S>> {
                let b = b.into();
                IterativeWorkspace::solve_owned(
                    a.cols(),
                    x0,
                    Method::$method.algorithm_id(),
                    |x, workspace| {
                        self.solve_into(
                            a,
                            b,
                            $( $preconditioning, )?
                            x,
                            workspace,
                            ExecutionContext::sequential_native(),
                        )
                    },
                )
            }
            /// Solves in place, starting from the values in `x`.
            ///
            /// Every product with `A` or `Aᴴ`, including those of residual
            /// verification and of the rounding-floor test, and every
            /// preconditioner application counts toward the evaluation limit. Stops on budgets, cancellation or a deadline
            /// return `Ok` as described in [Iterative solves](crate#iterative-solves).
            ///
            /// # Errors
            ///
            /// - The errors listed on
            ///   [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            ///   MINRES requires a square `A`; LSQR and LSMR require `b` of length
            ///   `a.rows()` and `x` of length `a.cols()`.
            /// - `InvalidInput` (`Inconsistent`) from MINRES if `A` is detectably
            ///   not Hermitian.
            pub fn solve_into<'b, 'scope, 'capability>(
                &self,
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
                $( $preconditioning: Preconditioning<'_, S>, )?
                mut x: VectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<IterativeSolveReport<S::Real>> {
                workspace.reject();
                let operator = Operator::$variant(a);
                self.engine.validate_shape(operator)?;
                let preconditioning = solver!(@inverse $( $preconditioning )?);
                preconditioning.validate_dimension(a.cols())?;
                let requirements = self.workspace_requirements(
                    &IterativeWorkspaceRequest::$request(a)?
                        .with_preconditioner(preconditioning)?,
                )?;
                workspace.run(
                    context.into().with_options(&self.engine.execution)?,
                    &Job {
                        algorithm: Method::$method.algorithm_id(),
                        limit: self.engine.limit(),
                        requirements,
                    },
                    b.into(),
                    &mut x,
                    |storage, b, x, kernels, solve| {
                        self.engine
                            .solve(preconditioning, operator, b, x, storage, kernels, solve)
                    },
                    IterativeProgress::of,
                )
            }
        }
    };
}
solver!(
    MinresSolver,
    r#"

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix, MinresSolver,
    Preconditioning, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = MinresSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- C. C. Paige and M. A. Saunders, "Solution of sparse indefinite systems of linear
  equations", *SIAM J. Numer. Anal.* 12(4), 1975."#,
    Minres,
    LinearOperator,
    Hermitian,
    operator,
    r#"Minimum residual method (MINRES) for Hermitian, possibly indefinite or singular, systems.

`A` must be square and Hermitian. MINRES minimizes `‖b − A xₖ‖₂` over the
Krylov subspace with a three-term Lanczos recurrence, so memory does not grow
with the iteration count. Each iteration costs one product with `A`, plus one
application of `M⁻¹` when a preconditioner is given. `M⁻¹` must be Hermitian
positive definite and the same at every application. Use [`CgSolver`](crate::CgSolver)
when `A` is positive definite and [`GmresSolver`](crate::GmresSolver) when it is
not Hermitian.

The true residual and `‖A r‖₂` are recomputed every 50 iterations and before
convergence is accepted, at the cost of two more products with `A`. A singular
consistent system converges like a regular one. For an inconsistent singular
system without preconditioning, the normal-residual test of
[`Self::with_normal_tolerance`] reports a least-squares solution. A
preconditioner that is detectably not Hermitian positive definite stops the
solve with
[`Breakdown(Preconditioner)`](crate::LinearSolveBreakdown::Preconditioner)."#,
    preconditioning
);
solver!(
    LsqrSolver,
    r#"

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, LsqrSolver, Matrix,
    Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = LsqrSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::adjoint(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- C. C. Paige and M. A. Saunders, "LSQR: An algorithm for sparse linear equations and
  sparse least squares", *ACM Trans. Math. Software* 8(1), 1982."#,
    Lsqr,
    AdjointOperator,
    Rectangular,
    adjoint,
    r#"LSQR for least-squares problems `min ‖b − A x‖₂` with a rectangular `A`.

`A` is an `m × n` [`AdjointOperator`](scioxide_core::AdjointOperator); `m < n`,
`m = n` and `m > n` are all accepted. LSQR applies the conjugate gradient method
to the normal equations `Aᴴ A x = Aᴴ b` through Golub–Kahan bidiagonalization,
without forming `Aᴴ A`. Each iteration costs one product with `A` and one with
`Aᴴ`. There is no preconditioner argument.

A consistent system converges on the true residual `‖b − A x‖₂`. An
inconsistent system stops with
[`LeastSquaresConverged`](crate::LinearSolveTermination::LeastSquaresConverged)
when `‖Aᴴ r‖₂` is small relative to `‖A‖·‖r‖₂`; see
[`Self::with_normal_tolerance`]. Both residuals are recomputed every 50 iterations
and before convergence is accepted. [`LsmrSolver`] solves the same problems; its
normal residual `‖Aᴴ r‖₂` decreases monotonically in exact arithmetic."#
);
solver!(
    LsmrSolver,
    r#"

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, LsmrSolver, Matrix,
    Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = LsmrSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::adjoint(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for sparse
  least-squares problems", *SIAM J. Sci. Comput.* 33(5), 2011."#,
    Lsmr,
    AdjointOperator,
    Rectangular,
    adjoint,
    r#"LSMR for least-squares problems `min ‖b − A x‖₂` with a rectangular `A`.

`A` is an `m × n` [`AdjointOperator`](scioxide_core::AdjointOperator); `m < n`,
`m = n` and `m > n` are all accepted. LSMR applies MINRES to the normal
equations `Aᴴ A x = Aᴴ b` through Golub–Kahan bidiagonalization, without
forming `Aᴴ A`. Each iteration costs one product with `A` and one with `Aᴴ`.
There is no preconditioner argument.

A consistent system converges on the true residual `‖b − A x‖₂`. An
inconsistent system stops with
[`LeastSquaresConverged`](crate::LinearSolveTermination::LeastSquaresConverged)
when `‖Aᴴ r‖₂` is small relative to `‖A‖·‖r‖₂`; see
[`Self::with_normal_tolerance`]. Both residuals are recomputed every 50 iterations
and before convergence is accepted. Compared with [`LsqrSolver`], the normal
residual `‖Aᴴ r‖₂` decreases monotonically in exact arithmetic."#
);

crate::traits::impl_iterative_solver!(MinresSolver, _plan, Method::Minres.algorithm_id());

fn divide<R: RealScalar>(a: R, b: R) -> Step<R> {
    if b == R::ZERO {
        Err(StepFault::Breakdown)
    } else {
        Ok(crate::checked::finite(
            a / b,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?)
    }
}

impl<S: NumericScalar> Engine<S> {
    fn normal_satisfied(
        &self,
        normal: S::Real,
        operator: S::Real,
        residual: S::Real,
    ) -> Result<bool> {
        use scioxide_core::Scaled;
        let (absolute, relative) = self
            .normal_tolerance
            .map_or((S::Real::ZERO, self.tolerance.relative()), |t| {
                (t.absolute(), t.relative())
            });
        let bound = Scaled::from_value(operator)?
            .mul(Scaled::from_value(residual)?)?
            .mul(Scaled::from_value(relative)?)?
            .add(Scaled::from_value(absolute)?)?;
        Ok(Scaled::from_value(normal)?.abs_le(bound))
    }
    /// Iteration allowance: `max_iterations`, capped by an explicit execution budget.
    fn limit(&self) -> Budget {
        self.execution.cap_iterations(self.max_iterations)
    }
    fn requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>> {
        IterativeRequirements::new(
            Shape {
                kind: StorageKind::Krylov,
                rows: request.rows(),
                cols: request.cols(),
                operator_scratch: request.operator_scratch(),
                preconditioner_scratch: request.preconditioner_scratch(),
                columns: 1,
                cycle: 0,
                retained: 0,
            },
            self.history.then_some(self.limit()),
            Some(request),
            self.execution.byte_limit(),
        )
    }
    fn validate_shape(&self, a: Operator<'_, S>) -> Result<()> {
        if self.method == Method::Minres {
            crate::checked::square(a.linear().rows(), a.linear().cols())?;
        }
        Ok(())
    }
    fn precondition(
        &self,
        preconditioner: Preconditioning<'_, S>,
        input: &Vector<S>,
        output: &mut Vector<S>,
        scratch: &mut [S],
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool> {
        super::support::precondition(
            preconditioner,
            false,
            input,
            output,
            scratch,
            kernels,
            context,
        )
    }
    fn positive_norm(
        &self,
        forms: &mut super::safety::HermitianForms<S::Real>,
        preconditioner: Preconditioning<'_, S>,
        r: &Vector<S>,
        z: &Vector<S>,
        kernels: crate::KernelContext<'_>,
        context: &scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<core::ops::ControlFlow<(), S::Real>> {
        use core::ops::ControlFlow;
        context.execution().check()?;
        if matches!(preconditioner, Preconditioning::None) {
            return Ok(ControlFlow::Continue(finite::fused_norm(
                r.as_slice(),
                kernels.dispatch(),
            )?));
        }
        let products =
            crate::kernel::recurrence::products(z.as_slice(), r.as_slice(), kernels.dispatch())?;
        let rn = products.right.norm(r.as_slice())?;
        if rn == S::Real::ZERO {
            return Ok(ControlFlow::Continue(rn));
        }
        let zn = products.left.norm(z.as_slice())?;
        let product = products.scaled(z.as_slice(), r.as_slice())?;
        Ok(match forms.positive_part(product, rn, zn)? {
            Some(real) => ControlFlow::Continue(real.sqrt()?.round()?),
            None => ControlFlow::Break(()),
        })
    }
    fn verify_residuals(
        &self,
        a: Operator<'_, S>,
        b: &Vector<S>,
        w: &mut Storage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<Option<VerifiedResiduals<S::Real>>> {
        // True diagnostics: `r = b - A x` and the normal-equation residual
        // `A^H r`, which is the meaningful stopping measure for inconsistent
        // least-squares problems where r itself cannot go to zero.
        let forward_len = a.linear().apply_workspace_len()?;
        let Some(residual) = super::support::true_residual_into_scoped(
            super::support::KrylovProblem {
                operator: a.linear(),
                rhs: b,
            },
            &w.x,
            &mut w.residual,
            &mut w.product,
            &mut w.scratch[..forward_len],
            kernels,
            context,
        )?
        else {
            return Ok(None);
        };
        let selected = w.best.observe(
            w.x.as_slice(),
            residual,
            context.counts().own().iterations(),
        );
        if selected {
            w.best_normal = None;
        }
        if !a.apply(
            true,
            &w.residual,
            &mut w.normal,
            &mut w.scratch,
            kernels,
            context,
        )? {
            return Ok(Some(VerifiedResiduals {
                residual,
                normal: None,
            }));
        }
        let normal = finite::fused_norm(w.normal.as_slice(), kernels.dispatch())?;
        if selected {
            w.best_normal = Some(normal);
        }
        Ok(Some(VerifiedResiduals {
            residual,
            normal: Some(normal),
        }))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    fn solve(
        &self,
        preconditioner: Preconditioning<'_, S>,
        a: Operator<'_, S>,
        b: &Vector<S>,
        x: &mut Vector<S>,
        w: &mut Storage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveReport<S::Real>> {
        let rhs_norm = finite::fused_norm(b.as_slice(), kernels.dispatch())?;
        let scale = super::residual_reference(rhs_norm);
        finite::copy(w.x.as_mut_slice(), x.as_slice())?;
        if let Some(history) = &mut w.history {
            history.clear();
        }
        w.best.residual = S::Real::INFINITY;
        w.best_normal = None;
        let initial = match self.verify_residuals(a, b, w, kernels, context) {
            Ok(initial) => initial,
            Err(error) if w.best.residual.is_finite() => {
                let termination = super::support::execution_stop(error)?;
                context.assess_accepted(super::checkpoint_trust(
                    self.tolerance.is_satisfied(w.best.residual, scale)?,
                ))?;
                if let Some(history) = &mut w.history {
                    history.push(w.best.residual);
                }
                return Ok(IterativeSolveReport::from_summary(solve_summary(
                    context,
                    termination,
                    TrueResidual::against(w.best.residual, scale),
                ))?
                .with_krylov(KrylovDiagnostics {
                    normal_residual: None,
                    condition_estimate: None,
                    rank_loss_detected: false,
                    restarts: 0,
                }));
            }
            Err(error) => return Err(error),
        };
        let Some(VerifiedResiduals {
            residual: initial_residual,
            normal: initial_normal,
        }) = initial
        else {
            return Err(evaluation_limit_error(context));
        };
        let Some(initial_normal) = initial_normal else {
            let satisfied = self.tolerance.is_satisfied(initial_residual, scale)?;
            context.assess_accepted(super::checkpoint_trust(satisfied))?;
            if self.history
                && let Some(history) = &mut w.history
            {
                history.push(initial_residual);
            }
            let termination = if satisfied {
                LinearSolveTermination::Converged
            } else {
                LinearSolveTermination::EvaluationLimit
            };
            let summary = solve_summary(
                context,
                termination,
                TrueResidual::against(initial_residual, scale),
            );
            return Ok(IterativeSolveReport::from_summary(summary)?.with_krylov(
                KrylovDiagnostics {
                    normal_residual: None,
                    condition_estimate: None,
                    rank_loss_detected: false,
                    restarts: 0,
                },
            ));
        };
        let mut verified = (initial_residual, initial_normal);
        let normal_scale = if verified.1 > S::Real::ZERO {
            verified.1
        } else {
            S::Real::ONE
        };
        finite::copy(w.accepted.as_mut_slice(), w.x.as_slice())?;
        context.assess_accepted(super::checkpoint_trust(
            self.tolerance.is_satisfied(verified.0, scale)?
                || (matches!(preconditioner, Preconditioning::None) && verified.1 == S::Real::ZERO),
        ))?;
        if self.history
            && let Some(history) = &mut w.history
        {
            history.push(verified.0);
        }
        let mut diagnostics = KrylovDiagnostics {
            normal_residual: None,
            condition_estimate: None,
            rank_loss_detected: false,
            restarts: 0,
        };
        let outcome = self.iterate_verified(
            preconditioner,
            a,
            b,
            w,
            kernels,
            context,
            KrylovProgress {
                verified: &mut verified,
                diagnostics: &mut diagnostics,
                scale,
                rhs_norm,
            },
        );
        let termination = match outcome {
            Ok(termination) => termination,
            Err(StepFault::Breakdown) => LinearSolveTermination::Breakdown(
                crate::LinearSolveBreakdown::RecurrenceDenominator,
            ),
            Err(StepFault::Error(error)) => match error.view() {
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
            verified.0 = w.best.residual;
        }
        if let Some(last) = w.history.as_mut().and_then(|samples| samples.last_mut()) {
            *last = verified.0;
        }
        // No output commit precedes successful report construction.
        diagnostics.normal_residual = (if termination.is_success() {
            Some(verified.1)
        } else {
            w.best_normal
        })
        .map(|normal| {
            ResidualReport::new(
                normal,
                NormSelection::L2,
                ResidualScaling::new(ResidualScalingRule::Reference, normal_scale)?,
            )
        })
        .transpose()?;
        let summary = solve_summary(
            context,
            termination,
            TrueResidual::against(verified.0, scale),
        );
        let report = IterativeSolveReport::from_summary(summary)?.with_krylov(diagnostics);
        finite::copy(x.as_mut_slice(), w.accepted.as_slice())?;
        Ok(report)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    fn iterate_verified(
        &self,
        preconditioner: Preconditioning<'_, S>,
        a: Operator<'_, S>,
        b: &Vector<S>,
        w: &mut Storage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
        progress: KrylovProgress<'_, S::Real>,
    ) -> Step<LinearSolveTermination> {
        let KrylovProgress {
            verified,
            diagnostics,
            scale,
            rhs_norm,
        } = progress;
        let mut state = Recurrence::new(self.method, w.cols);
        let floor = super::safety::ResidualFloor::new(rhs_norm);
        let mut initialized = false;
        let mut pending_restart = false;
        loop {
            if self.tolerance.is_satisfied(verified.0, scale)? {
                return Ok(LinearSolveTermination::Converged);
            }
            // An exactly stationary initial residual is already a least-squares solution.
            if matches!(preconditioner, Preconditioning::None) && verified.1 == S::Real::ZERO {
                return Ok(LinearSolveTermination::LeastSquaresConverged);
            }
            if let Some(reason) = context.stop_reason()? {
                return Ok(LinearSolveTermination::ExecutionStopped(reason));
            }
            if context.counts().own().iterations() >= super::support::iteration_limit(context) {
                return Ok(LinearSolveTermination::IterationLimit);
            }
            if !initialized {
                if !self.initialize(preconditioner, w, &mut state, kernels, context)? {
                    return Ok(LinearSolveTermination::EvaluationLimit);
                }
                if pending_restart {
                    diagnostics.restarts += 1;
                    pending_restart = false;
                }
                initialized = true;
            }
            if let Some(kind) = state.breakdown {
                return Ok(LinearSolveTermination::Breakdown(kind));
            }
            if !context.record_iteration()?.is_granted() {
                return Ok(LinearSolveTermination::IterationLimit);
            }
            state.unchanged = false;
            let completed = match self.method {
                Method::Minres => {
                    self.minres_step(preconditioner, a, w, &mut state, kernels, context)?
                }
                Method::Lsqr | Method::Lsmr => {
                    self.least_squares_step(a, w, &mut state, kernels, context)?
                }
            };
            if !completed {
                return Ok(LinearSolveTermination::EvaluationLimit);
            }
            if let Some(kind) = state.breakdown {
                return Ok(LinearSolveTermination::Breakdown(kind));
            }
            let reciprocal = state.reciprocal()?;
            diagnostics.rank_loss_detected |= state.rank_loss;
            diagnostics.condition_estimate = Some(ConditionEstimate::new(
                reciprocal,
                EstimateReliability::ModelBased,
            )?);
            let estimate = if !matches!(preconditioner, Preconditioning::None) {
                finite::fused_norm(w.minres_residual.as_slice(), kernels.dispatch())?
            } else {
                state.phibar.abs()
            };
            let candidate = state.held
                || self.tolerance.is_satisfied(estimate, scale)?
                || (matches!(preconditioner, Preconditioning::None)
                    && self.normal_satisfied(
                        state.normal_estimate,
                        state.operator_norm,
                        estimate,
                    )?);
            let periodic = super::safety::should_recompute_true_residual(state.steps);
            let at_limit =
                context.counts().own().iterations() >= super::support::iteration_limit(context);
            let checkpoint = candidate
                || periodic
                || state.rank_loss
                || state.unchanged
                || reciprocal < self.condition_limit
                || at_limit;
            if !checkpoint {
                if let Some(history) = &mut w.history {
                    history.push(estimate);
                }
                continue;
            }
            let best_before = w.best.residual;
            let Some(VerifiedResiduals {
                residual,
                normal: Some(normal),
            }) = self.verify_residuals(a, b, w, kernels, context)?
            else {
                return Ok(LinearSolveTermination::EvaluationLimit);
            };
            let unchanged = w.x.as_slice() == w.accepted.as_slice();
            *verified = (residual, normal);
            finite::copy(w.accepted.as_mut_slice(), w.x.as_slice())?;
            let converged = self.tolerance.is_satisfied(residual, scale)?;
            let least_squares = matches!(preconditioner, Preconditioning::None)
                && self.normal_satisfied(normal, state.operator_norm, residual)?;
            context.commit_accepted(super::checkpoint_trust(converged || least_squares))?;
            if let Some(history) = &mut w.history {
                history.push(residual);
            }
            if converged {
                return Ok(LinearSolveTermination::Converged);
            }
            if least_squares {
                return Ok(LinearSolveTermination::LeastSquaresConverged);
            }
            if reciprocal < self.condition_limit {
                return Ok(LinearSolveTermination::ConditionLimit);
            }
            if unchanged || state.unchanged {
                return Ok(LinearSolveTermination::Stagnation);
            }
            // The next Lanczos or Golub–Kahan vector and the operator image are
            // recomputed by the next step, so they hold the floor probe.
            if !at_limit && super::safety::floor_test_due(residual, best_before) {
                match floor.reached(
                    a.linear(),
                    w.best.solution.as_slice(),
                    w.best.residual,
                    super::safety::FloorProbe {
                        input: w.next_v.as_mut_slice(),
                        image: w.product.as_mut_slice(),
                        workspace: &mut w.scratch[..a.linear().apply_workspace_len()?],
                    },
                    kernels,
                    context,
                )? {
                    None => return Ok(LinearSolveTermination::EvaluationLimit),
                    Some(true) => return Ok(LinearSolveTermination::Stagnation),
                    Some(false) => {}
                }
            }
            // Verification does not alter the Lanczos/Golub–Kahan recurrence.
            // Only a false convergence claim, a held update or a lost
            // direction starts a new segment.
            state.unverified = false;
            if candidate || state.rank_loss {
                let earlier_norm = state.operator_norm.max(state.earlier_norm);
                state = Recurrence::new(self.method, w.cols);
                state.earlier_norm = earlier_norm;
                initialized = false;
                pending_restart = true;
            }
        }
    }

    fn initialize(
        &self,
        preconditioner: Preconditioning<'_, S>,
        w: &mut Storage<S>,
        state: &mut Recurrence<S::Real>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        finite::copy(w.x.as_mut_slice(), w.accepted.as_slice())?;
        finite::copy(w.u.as_mut_slice(), w.residual.as_slice())?;
        w.w.view_mut().fill(S::ZERO);
        w.previous_w.view_mut().fill(S::ZERO);
        w.hbar.view_mut().fill(S::ZERO);
        w.previous_u.view_mut().fill(S::ZERO);
        if self.method == Method::Minres {
            finite::copy(w.minres_residual.as_mut_slice(), w.residual.as_slice())?;
            for image in &mut w.minres_images {
                image.view_mut().fill(S::ZERO);
            }
            if !matches!(preconditioner, Preconditioning::None)
                && !self.precondition(
                    preconditioner,
                    &w.u,
                    &mut w.v,
                    &mut w.preconditioner_scratch,
                    kernels,
                    context,
                )?
            {
                return Ok(false);
            }
            state.beta = match self.positive_norm(
                &mut state.preconditioner_forms,
                preconditioner,
                &w.u,
                &w.v,
                kernels,
                context,
            )? {
                core::ops::ControlFlow::Continue(value) => value,
                core::ops::ControlFlow::Break(()) => {
                    state.breakdown = Some(crate::LinearSolveBreakdown::Preconditioner);
                    return Ok(true);
                }
            };
            state.phibar = state.beta;
        } else {
            state.beta = finite::fused_norm(w.u.as_slice(), kernels.dispatch())?;
            if state.beta == S::Real::ZERO {
                w.u.view_mut().fill(S::ZERO);
                Ok(())
            } else {
                super::safety::normalized(finite::normalize(
                    w.u.as_mut_slice(),
                    state.beta,
                    kernels.dispatch(),
                ))
            }?;
            // Both bidiagonal methods reuse the independently computed Aᴴr.
            finite::copy(w.v.as_mut_slice(), w.normal.as_slice())?;
            let normal = finite::fused_norm(w.v.as_slice(), kernels.dispatch())?;
            state.alpha = divide(normal, state.beta)?;
            if normal != S::Real::ZERO {
                super::safety::normalized(finite::normalize(
                    w.v.as_mut_slice(),
                    normal,
                    kernels.dispatch(),
                ))?;
            }
            finite::copy(w.w.as_mut_slice(), w.v.as_slice())?;
            state.alphabar = state.alpha;
            state.phibar = state.beta;
            state.betadd = state.beta;
            state.zetabar = crate::checked::finite(
                state.alpha * state.beta,
                scioxide_core::NonFiniteOrigin::Intermediate,
                "non-finite recurrence scalar",
            )?;
        }
        Ok(true)
    }
    fn minres_step(
        &self,
        preconditioner: Preconditioning<'_, S>,
        a: Operator<'_, S>,
        w: &mut Storage<S>,
        s: &mut Recurrence<S::Real>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        // One step of preconditioned MINRES (Paige-Saunders): a Lanczos step
        // produces `alpha` and the next `beta`; the previous two Givens
        // rotations are applied to the new tridiagonal column (`delta`,
        // `epsilon`, `gbar`), a new rotation zeroes `beta`, and the solution
        // is updated along the three-term direction `hbar`. `phibar` tracks
        // the residual norm of the projected least-squares problem. A zero
        // `gamma` or a `beta` at roundoff level signals rank loss.
        finite::copy(
            w.next_v.as_mut_slice(),
            if matches!(preconditioner, Preconditioning::None) {
                w.u.as_slice()
            } else {
                w.v.as_slice()
            },
        )?;
        if s.beta == S::Real::ZERO {
            w.next_v.view_mut().fill(S::ZERO);
            Ok(())
        } else {
            super::safety::normalized(finite::normalize(
                w.next_v.as_mut_slice(),
                s.beta,
                kernels.dispatch(),
            ))
        }?;
        if !a.apply(
            false,
            &w.next_v,
            &mut w.product,
            &mut w.scratch,
            kernels,
            context,
        )? {
            return Ok(false);
        }
        if !matches!(preconditioner, Preconditioning::None) {
            finite::copy(w.minres_images[2].as_mut_slice(), w.product.as_slice())?;
        }
        if s.steps > 0 {
            finite::update(
                w.product.as_mut_slice(),
                S::from_real(-divide(s.beta, s.old_beta)?),
                w.previous_u.as_slice(),
                S::ONE,
                kernels.dispatch(),
            )?;
        }
        let products = crate::kernel::recurrence::products(
            w.product.as_slice(),
            w.next_v.as_slice(),
            kernels.dispatch(),
        )?;
        // The Lanczos recurrence needs a real `vᴴAv`; an imaginary part above
        // the operator rounding bound means `A` or `M⁻¹` is not Hermitian.
        let alpha = s
            .operator_forms
            .real_part(
                products.scaled(w.product.as_slice(), w.next_v.as_slice())?,
                products.right.norm(w.next_v.as_slice())?,
                products.left.norm(w.product.as_slice())?,
            )?
            .ok_or_else(|| {
                crate::checked::inconsistent(
                    scioxide_core::argument!("a"),
                    "operator violated the Hermitian recurrence",
                )
            })?
            .round()?;
        finite::update(
            w.product.as_mut_slice(),
            S::from_real(-divide(alpha, s.beta)?),
            w.u.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?;
        core::mem::swap(&mut w.previous_u, &mut w.u);
        core::mem::swap(&mut w.u, &mut w.product);
        if !matches!(preconditioner, Preconditioning::None)
            && !self.precondition(
                preconditioner,
                &w.u,
                &mut w.v,
                &mut w.preconditioner_scratch,
                kernels,
                context,
            )?
        {
            return Ok(false);
        }
        s.old_beta = s.beta;
        s.beta = match self.positive_norm(
            &mut s.preconditioner_forms,
            preconditioner,
            &w.u,
            &w.v,
            kernels,
            context,
        )? {
            core::ops::ControlFlow::Continue(value) => value,
            core::ops::ControlFlow::Break(()) => {
                s.breakdown = Some(crate::LinearSolveBreakdown::Preconditioner);
                return Ok(true);
            }
        };
        s.operator_norm =
            crate::checked::finite_candidate(s.operator_norm.hypot(alpha).hypot(s.beta))?;
        let old_epsilon = s.epsilon;
        let delta = crate::checked::finite(
            s.c * s.dbar + s.s * alpha,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        let gbar = crate::checked::finite(
            s.s * s.dbar - s.c * alpha,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        s.epsilon = crate::checked::finite(
            s.s * s.beta,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        s.dbar = crate::checked::finite(
            -s.c * s.beta,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        // Paige–Saunders estimate of ||A r|| for the preceding projected
        // residual. This lagged estimate only requests a true-residual check;
        // it never establishes convergence of the new iterate by itself.
        s.normal_estimate = crate::checked::finite_candidate(s.phibar.abs() * gbar.hypot(s.dbar))?;
        // The estimate describes the iterate before this step. When it already
        // passes the normal-residual test, that iterate is verified instead of
        // taking a step whose pivot `gamma` vanishes for an inconsistent
        // singular system.
        if s.unverified
            && matches!(preconditioner, Preconditioning::None)
            && self.normal_satisfied(s.normal_estimate, s.operator_norm, s.phibar.abs())?
        {
            s.held = true;
            return Ok(true);
        }
        let (c, sn, gamma) = crate::rotation::Rotation::generate(gbar, s.beta).map(|(g, r)| {
            if r < S::Real::ZERO {
                (-g.c, -g.s, -r)
            } else {
                (g.c, g.s, r)
            }
        })?;
        // A pivot at the rounding level of `‖T‖` makes `phi / gamma` pure
        // rounding error, which a singular `A` turns into a null-space
        // component. After a restart from a least-squares solution the new
        // segment sees only rounding noise, so the scale spans all segments.
        if gamma <= dimension_epsilon::<S::Real>(w.cols)? * s.operator_norm.max(s.earlier_norm) {
            s.rank_loss = true;
            return Ok(true);
        }
        s.c = c;
        s.s = sn;
        let phi = crate::checked::finite(
            c * s.phibar,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        s.phibar = crate::checked::finite(
            sn * s.phibar,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        finite::copy(w.hbar.as_mut_slice(), w.next_v.as_slice())?;
        finite::update(
            w.hbar.as_mut_slice(),
            S::from_real(-old_epsilon),
            w.previous_w.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?;
        finite::update(
            w.hbar.as_mut_slice(),
            S::from_real(-delta),
            w.w.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?;
        super::safety::normalized(finite::normalize(
            w.hbar.as_mut_slice(),
            gamma,
            kernels.dispatch(),
        ))?;
        s.unchanged = !finite::update(
            w.x.as_mut_slice(),
            S::from_real(phi),
            w.hbar.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?
        .changed;
        s.unverified |= !s.unchanged;
        if !matches!(preconditioner, Preconditioning::None) {
            let [current, previous, next] = &mut w.minres_images;
            finite::update(
                next.as_mut_slice(),
                S::from_real(-old_epsilon),
                previous.as_slice(),
                S::ONE,
                kernels.dispatch(),
            )?;
            finite::update(
                next.as_mut_slice(),
                S::from_real(-delta),
                current.as_slice(),
                S::ONE,
                kernels.dispatch(),
            )?;
            super::safety::normalized(finite::normalize(
                next.as_mut_slice(),
                gamma,
                kernels.dispatch(),
            ))?;
            finite::update(
                w.minres_residual.as_mut_slice(),
                S::from_real(-phi),
                next.as_slice(),
                S::ONE,
                kernels.dispatch(),
            )?;
            core::mem::swap(previous, current);
            core::mem::swap(current, next);
        }
        core::mem::swap(&mut w.previous_w, &mut w.w);
        core::mem::swap(&mut w.w, &mut w.hbar);
        s.observe(gamma)?;
        s.steps += 1;
        s.rank_loss = s.beta <= dimension_epsilon::<S::Real>(w.cols)? * alpha.abs().max(s.old_beta);
        Ok(true)
    }
    fn least_squares_step(
        &self,
        a: Operator<'_, S>,
        w: &mut Storage<S>,
        s: &mut Recurrence<S::Real>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        // Golub-Kahan bidiagonalization step shared by LSQR and LSMR:
        // `beta u = A v - alpha u`, `alpha v = A^H u - beta v`. The method-
        // specific rotation update then advances x. Tiny `alpha` or `beta`
        // (relative to their predecessors) means the bidiagonalization has
        // hit an invariant subspace or numerical rank deficiency.
        if !a.apply(
            false,
            &w.v,
            &mut w.product,
            &mut w.scratch,
            kernels,
            context,
        )? {
            return Ok(false);
        }
        let u_moments = finite::update(
            w.product.as_mut_slice(),
            S::from_real(-s.alpha),
            w.u.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?
        .moments;
        core::mem::swap(&mut w.u, &mut w.product);
        let old_alpha = s.alpha;
        s.beta = u_moments.norm(w.u.as_slice())?;
        if s.beta == S::Real::ZERO {
            w.u.view_mut().fill(S::ZERO);
            Ok(())
        } else {
            super::safety::normalized(crate::kernel::recurrence::normalize(
                w.u.as_mut_slice(),
                s.beta,
                u_moments.maximum,
                kernels.dispatch(),
            ))
        }?;
        if !a.apply(true, &w.u, &mut w.next_v, &mut w.scratch, kernels, context)? {
            return Ok(false);
        }
        let v_moments = finite::update(
            w.next_v.as_mut_slice(),
            S::from_real(-s.beta),
            w.v.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?
        .moments;
        s.alpha = v_moments.norm(w.next_v.as_slice())?;
        if s.alpha == S::Real::ZERO {
            w.next_v.view_mut().fill(S::ZERO);
            Ok(())
        } else {
            super::safety::normalized(crate::kernel::recurrence::normalize(
                w.next_v.as_mut_slice(),
                s.alpha,
                v_moments.maximum,
                kernels.dispatch(),
            ))
        }?;
        core::mem::swap(&mut w.v, &mut w.next_v);
        match self.method {
            Method::Lsqr => {
                s.operator_norm = crate::checked::finite(
                    s.operator_norm.hypot(old_alpha).hypot(s.beta),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                    "non-finite recurrence scalar",
                )?;
                let direction_norm = finite::fused_norm(w.w.as_slice(), kernels.dispatch())?;
                lsqr_update(w, s, direction_norm, kernels)?;
            }
            Method::Lsmr => {
                s.operator_norm = crate::checked::finite_candidate(
                    s.operator_norm.hypot(old_alpha).hypot(s.beta),
                )?;
                lsmr_update(w, s, kernels)?;
            }
            Method::Minres => {
                return Err(StepFault::Error(crate::checked::inconsistent(
                    scioxide_core::argument!("self"),
                    "invalid bidiagonalization method",
                )));
            }
        }
        s.steps += 1;
        let threshold = dimension_epsilon::<S::Real>(w.rows.max(w.cols))? * old_alpha.max(s.beta);
        s.rank_loss |= s.alpha <= threshold || s.beta <= threshold;
        Ok(true)
    }
}

#[derive(Debug)]
struct Recurrence<R: RealScalar> {
    alpha: R,
    beta: R,
    old_beta: R,
    alphabar: R,
    phibar: R,
    normal_estimate: R,
    /// The last step applied an update that left `x` unchanged.
    unchanged: bool,
    /// `x` changed since its last verification.
    unverified: bool,
    /// The last step kept `x` for verification instead of updating it.
    held: bool,
    c: R,
    s: R,
    dbar: R,
    epsilon: R,
    rho: R,
    rhobar: R,
    cbar: R,
    sbar: R,
    zetabar: R,
    zeta: R,
    betadd: R,
    betad: R,
    rhod: R,
    tautilde: R,
    thetatilde: R,
    operator_norm: R,
    /// Largest `operator_norm` of the earlier segments of the solve.
    earlier_norm: R,
    inverse_norm: R,
    min_diagonal: R,
    max_diagonal: R,
    steps: usize,
    rank_loss: bool,
    breakdown: Option<crate::LinearSolveBreakdown>,
    operator_forms: super::safety::HermitianForms<R>,
    preconditioner_forms: super::safety::HermitianForms<R>,
}
impl<R: RealScalar> Recurrence<R> {
    fn new(method: Method, dimension: usize) -> Self {
        Self {
            alpha: R::ZERO,
            beta: R::ZERO,
            old_beta: R::ZERO,
            alphabar: R::ZERO,
            phibar: R::ZERO,
            normal_estimate: R::ZERO,
            unchanged: false,
            unverified: false,
            held: false,
            c: if method == Method::Minres {
                -R::ONE
            } else {
                R::ONE
            },
            s: R::ZERO,
            dbar: R::ZERO,
            epsilon: R::ZERO,
            rho: R::ONE,
            rhobar: R::ONE,
            cbar: R::ONE,
            sbar: R::ZERO,
            zetabar: R::ZERO,
            zeta: R::ZERO,
            betadd: R::ZERO,
            betad: R::ZERO,
            rhod: R::ONE,
            tautilde: R::ZERO,
            thetatilde: R::ZERO,
            operator_norm: R::ZERO,
            earlier_norm: R::ZERO,
            inverse_norm: R::ZERO,
            min_diagonal: R::ZERO,
            max_diagonal: R::ZERO,
            steps: 0,
            rank_loss: false,
            breakdown: None,
            operator_forms: super::safety::HermitianForms::new(dimension),
            preconditioner_forms: super::safety::HermitianForms::new(dimension),
        }
    }
    fn observe(&mut self, diagonal: R) -> Result<()> {
        let value = crate::checked::finite(
            diagonal.abs(),
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite recurrence scalar",
        )?;
        if self.steps == 0 {
            self.min_diagonal = value;
            self.max_diagonal = value;
        } else {
            self.min_diagonal = self.min_diagonal.min(value);
            self.max_diagonal = self.max_diagonal.max(value);
        }
        Ok(())
    }
    fn reciprocal(&self) -> Result<R> {
        if self.max_diagonal == R::ZERO {
            Ok(R::ZERO)
        } else {
            crate::checked::finite(
                self.min_diagonal / self.max_diagonal,
                scioxide_core::NonFiniteOrigin::Intermediate,
                "non-finite recurrence scalar",
            )
        }
    }
}

/// QR update for Paige–Saunders LSQR, after one Golub–Kahan step.
///
fn lsqr_update<S: NumericScalar>(
    w: &mut Storage<S>,
    s: &mut Recurrence<S::Real>,
    direction_norm: S::Real,
    kernels: crate::KernelContext<'_>,
) -> Step<()> {
    let (c, sn, rho) = crate::rotation::Rotation::generate(s.alphabar, s.beta).map(|(g, r)| {
        if r < S::Real::ZERO {
            (-g.c, -g.s, -r)
        } else {
            (g.c, g.s, r)
        }
    })?;
    if rho == S::Real::ZERO {
        s.rank_loss = true;
        return Ok(());
    }
    let theta = crate::checked::finite(
        sn * s.alpha,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    s.alphabar = crate::checked::finite(
        -c * s.alpha,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    let phi = crate::checked::finite(
        c * s.phibar,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    s.phibar = crate::checked::finite(
        sn * s.phibar,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    // LSQR's normal-residual estimate alpha_{k+1} * |s_k phi_k|.
    s.normal_estimate = crate::checked::finite_candidate(s.alpha * (sn * phi).abs())?;
    s.unchanged = !finite::update(
        w.x.as_mut_slice(),
        S::from_real(divide(phi, rho)?),
        w.w.as_slice(),
        S::ONE,
        kernels.dispatch(),
    )?
    .changed;
    finite::update(
        w.w.as_mut_slice(),
        S::ONE,
        w.v.as_slice(),
        S::from_real(-divide(theta, rho)?),
        kernels.dispatch(),
    )?;
    // Paige-Saunders condition estimate: `||A||_F` accumulates from the
    // bidiagonal entries and `||A^+||` from the scaled direction norms; the
    // reciprocal of their product is stored as a diagonal-ratio surrogate.
    s.inverse_norm = crate::checked::finite(
        s.inverse_norm.hypot(divide(direction_norm, rho)?),
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    // Store the reciprocal norm-product estimate without forming an overflowing product.
    let reciprocal = if s.operator_norm >= S::Real::ONE {
        divide(divide(S::Real::ONE, s.operator_norm)?, s.inverse_norm)?
    } else {
        divide(divide(S::Real::ONE, s.inverse_norm)?, s.operator_norm)?
    };
    s.min_diagonal = reciprocal.min(S::Real::ONE);
    s.max_diagonal = S::Real::ONE;
    Ok(())
}

/// Two-rotation update for Fong–Saunders LSMR, after one Golub–Kahan step.
///
fn lsmr_update<S: NumericScalar>(
    w: &mut Storage<S>,
    s: &mut Recurrence<S::Real>,
    kernels: crate::KernelContext<'_>,
) -> Step<()> {
    let old_rho = s.rho;
    let old_rhobar = s.rhobar;
    let (c, sn, rho) = crate::rotation::Rotation::generate(s.alphabar, s.beta).map(|(g, r)| {
        if r < S::Real::ZERO {
            (-g.c, -g.s, -r)
        } else {
            (g.c, g.s, r)
        }
    })?;
    s.rho = rho;
    let theta = crate::checked::finite(
        sn * s.alpha,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    s.alphabar = crate::checked::finite(
        c * s.alpha,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    let thetabar = crate::checked::finite(
        s.sbar * rho,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    let (cbar, sbar, rhobar) =
        crate::rotation::Rotation::generate(s.cbar * rho, theta).map(|(g, r)| {
            if r < S::Real::ZERO {
                (-g.c, -g.s, -r)
            } else {
                (g.c, g.s, r)
            }
        })?;
    s.cbar = cbar;
    s.sbar = sbar;
    s.rhobar = rhobar;
    if rho == S::Real::ZERO || rhobar == S::Real::ZERO {
        s.rank_loss = true;
        return Ok(());
    }
    let zeta = crate::checked::finite(
        cbar * s.zetabar,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    s.zetabar = crate::checked::finite(
        -sbar * s.zetabar,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    let factor = crate::checked::finite(
        divide(thetabar, old_rhobar)? * divide(rho, old_rho)?,
        scioxide_core::NonFiniteOrigin::Intermediate,
        "non-finite recurrence scalar",
    )?;
    finite::update(
        w.hbar.as_mut_slice(),
        S::ONE,
        w.w.as_slice(),
        S::from_real(-factor),
        kernels.dispatch(),
    )?;
    s.unchanged = !finite::update(
        w.x.as_mut_slice(),
        S::from_real(divide(divide(zeta, rho)?, rhobar)?),
        w.hbar.as_slice(),
        S::ONE,
        kernels.dispatch(),
    )?
    .changed;
    finite::update(
        w.w.as_mut_slice(),
        S::ONE,
        w.v.as_slice(),
        S::from_real(-divide(theta, rho)?),
        kernels.dispatch(),
    )?;
    // Fong–Saunders (2011), §3.3: rotate the residual RHS alongside the
    // triangular factors. Hypot avoids squaring small or large residuals.
    let betahat = c * s.betadd;
    s.betadd = -sn * s.betadd;
    let old_theta = s.thetatilde;
    let (rotation, rhotilde) = crate::rotation::Rotation::generate(s.rhod, thetabar)?;
    s.thetatilde = rotation.s * rhobar;
    s.rhod = rotation.c * rhobar;
    s.betad = -rotation.s * s.betad + rotation.c * betahat;
    s.tautilde = divide(s.zeta - old_theta * s.tautilde, rhotilde)?;
    let taud = divide(zeta - s.thetatilde * s.tautilde, s.rhod)?;
    s.zeta = zeta;
    s.phibar = crate::checked::finite_candidate((s.betad - taud).hypot(s.betadd))?;
    s.normal_estimate = s.zetabar.abs();
    Ok(s.observe(rhobar)?)
}
