//! Sylvester and Lyapunov equations solved by the Bartels–Stewart method.
//!
//! Plans compute the Schur forms of the coefficients once. Each solve transforms
//! the right-hand side, solves the quasi-triangular equation by block
//! substitution, transforms back and checks the residual of the original equation.
use super::blocked;
use super::blocked::Products;
use crate::DirectSolveTermination;
use crate::kernel::frames::{Framed, Grid, Input, WIDE};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::take_mut;
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    Estimation, LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace, Matrix, MatrixView, MatrixViewMut, SchurOptions,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    Budget, Dimensions, ExecutionContext, ExecutionPolicy, RealScalar, ResidualTolerance,
    SciOxideError, SolveContext, Tolerance,
};
use scioxide_core::{Scaled, ScaledValue};

/// Options of a [`SylvesterPlan`] or [`LyapunovPlan`].
///
/// A solve is accepted when the residual of the original equation satisfies
///
/// ```text
/// ‖R‖_F ≤ atol + rtol · (‖T₁‖_F + ‖T₂‖_F + ‖T₃‖_F),
/// ```
///
/// where `R = T₁ + T₂ + T₃` and `T₁`, `T₂`, `T₃` are the three terms of the
/// equation with the computed solution `X`, for example `A X`, `s X B` and `−C`
/// for the Sylvester equation.
///
/// Defaults: `atol = rtol = √ε` ([`Tolerance::from_sqrt_epsilon`]), the separation
/// estimate is computed ([`Estimation::Estimate`]), and a solve may use at most
/// 1 000 000 work units unless
/// [`LinalgExecutionOptions::with_iteration_budget`] sets another limit. Work
/// units count block operations on tiles of order at most 32 and the 1×1 or 2×2
/// diagonal-block solves of the substitution. The Schur reduction of a plan
/// uses the configured execution policy, and an explicit iteration budget also
/// limits its QR iteration, which otherwise may take `30 max(n, 1)` steps and
/// typically takes about two per eigenvalue: a budget below that fails the
/// plan construction. Block substitution is sequential.
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_linalg::{Estimation, LinalgExecutionOptions, MatrixEquationOptions};
/// let budget = Budget::ONE.saturating_add(99);
/// let options = MatrixEquationOptions::<f64>::new()
///     .with_separation(Estimation::Skip)
///     .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(budget));
/// assert_eq!(options.separation(), Estimation::Skip);
/// assert_eq!(options.execution_options().iteration_budget().map(Budget::get), Some(100));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MatrixEquationOptions<R: RealScalar> {
    execution: LinalgExecutionOptions,
    tolerance: ResidualTolerance<R>,
    separation: Estimation,
}
impl<R: RealScalar> Default for MatrixEquationOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> MatrixEquationOptions<R> {
    /// Sets the execution options, including the work limit (iteration budget).
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            tolerance: ResidualTolerance::from_tolerance(Tolerance::from_sqrt_epsilon()),
            separation: Estimation::Estimate,
        }
    }
    /// Sets the residual tolerance `atol`, `rtol` of the acceptance test.
    pub const fn with_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Selects whether solves estimate the separation reported by
    /// [`MatrixEquationReport::separation`].
    ///
    /// [`Estimation::Skip`] saves the estimate's inverse applications and one
    /// solution-sized workspace buffer.
    pub const fn with_separation(mut self, separation: Estimation) -> Self {
        self.separation = separation;
        self
    }
    /// Returns the residual tolerance.
    pub const fn tolerance(self) -> ResidualTolerance<R> {
        self.tolerance
    }
    /// Returns whether the separation is estimated.
    pub const fn separation(self) -> Estimation {
        self.separation
    }
    /// Maximum number of work units per solve.
    pub(crate) const fn work_limit(self) -> Budget {
        match self.execution.iteration_budget() {
            Some(limit) => limit,
            None => scioxide_core::Budget::ONE.saturating_add(999999),
        }
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }
}

/// Sign `s` of the Sylvester equation `A X + s X B = C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SylvesterSign {
    /// `A X + X B = C`.
    Plus,
    /// `A X − X B = C`.
    Minus,
}

/// Form of the Lyapunov equation solved by a [`LyapunovPlan`].
///
/// `Aᴴ` is the conjugate transpose of `A` (the transpose for real `A`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LyapunovKind {
    /// Continuous-time equation `A X + X Aᴴ = Q`.
    Continuous,
    /// Discrete-time (Stein) equation `A X Aᴴ − X + Q = 0`.
    Discrete,
}

/// Diagnostics of one Sylvester or Lyapunov solve.
///
/// The residual is computed from the original `A`, `B` and right-hand side,
/// not from their Schur forms.
#[derive(Debug, Clone, Copy)]
pub struct MatrixEquationReport<R: RealScalar> {
    /// Acceptance status.
    pub(crate) termination: DirectSolveTermination,
    /// Frobenius norm of the original-equation residual.
    pub(crate) residual: R,
    /// Residual divided by the sum of Frobenius norms of equation terms.
    pub(crate) backward_error: R,
    /// Maximum entrywise symmetry defect before optional Hermitian projection.
    pub(crate) symmetry_defect: Option<R>,
    /// Reciprocal inverse one-norm estimate in Schur coordinates.
    pub(crate) separation: Option<R>,
    /// Work units used by transforms, substitution and the residual check.
    pub(crate) work: usize,
}

impl<R: RealScalar> MatrixEquationReport<R> {
    /// Returns [`DirectSolveTermination::Accepted`] if the residual test passed,
    /// and [`DirectSolveTermination::ResidualRejected`] otherwise.
    pub const fn termination(&self) -> DirectSolveTermination {
        self.termination
    }

    /// Returns the Frobenius norm `‖R‖_F` of the original-equation residual.
    pub const fn residual(&self) -> R {
        self.residual
    }

    /// Returns `‖R‖_F / (‖T₁‖_F + ‖T₂‖_F + ‖T₃‖_F)`, or zero when all terms vanish.
    ///
    /// The terms are defined on [`MatrixEquationOptions`].
    pub const fn backward_error(&self) -> R {
        self.backward_error
    }

    /// Returns `maxᵢⱼ |xᵢⱼ − conj(xⱼᵢ)|` of the computed Lyapunov solution, or
    /// `None` for a Sylvester equation.
    ///
    /// When the right-hand side `Q` is exactly Hermitian, the solution is then
    /// replaced by its Hermitian part `(X + Xᴴ)/2` before the residual check.
    /// The defect is measured before this projection.
    pub const fn symmetry_defect(&self) -> Option<R> {
        self.symmetry_defect
    }

    /// Returns an estimate of `1/‖L⁻¹‖₁`, where `L` is the linear operator of
    /// the equation in Schur coordinates acting on `vec(X)`.
    ///
    /// For the Sylvester equation, `L(Y) = T_A Y + s Y T_B` with the Schur
    /// factors `T_A`, `T_B`, and the value is a one-norm analogue of the
    /// separation `sep(A, −s B)`. A small value means that the solution is
    /// sensitive to perturbations of the data. The estimate comes from Hager's
    /// method as refined by Higham (exact for at most 4 unknowns) and is not a
    /// bound. It is `None` when estimation is disabled ([`Estimation::Skip`]) or
    /// the equation is empty, and zero when the estimator did not complete.
    pub const fn separation(&self) -> Option<R> {
        self.separation
    }

    /// Returns the number of work units used, including transforms, substitution,
    /// the separation estimate and the residual check.
    pub const fn work(&self) -> usize {
        self.work
    }
}

/// Solution of a Sylvester or Lyapunov equation and its diagnostics.
///
/// The solution is available only when the residual test accepted it
/// (see [direct solves](crate#direct-solves)).
#[derive(Debug)]
pub struct MatrixEquationResult<S: NumericScalar> {
    /// Solution in the input scalar domain.
    pub(crate) solution: Matrix<S>,
    /// Numerical diagnostics and acceptance.
    pub(crate) report: MatrixEquationReport<S::Real>,
}

impl<S: NumericScalar> MatrixEquationResult<S> {
    /// Returns the accepted solution, or `None` if it was rejected, and the report.
    pub fn into_parts(self) -> (Option<Matrix<S>>, MatrixEquationReport<S::Real>) {
        (
            self.report
                .termination
                .is_success()
                .then_some(self.solution),
            self.report,
        )
    }

    /// Returns the accepted solution, or `None` if it was rejected.
    pub fn solution(&self) -> Option<&Matrix<S>> {
        self.report
            .termination
            .is_success()
            .then_some(&self.solution)
    }

    /// Returns the accepted solution, or the error of
    /// [`DirectSolveTermination::require_accepted`] for a rejected one.
    ///
    /// # Errors
    ///
    /// `NumericalInstability` (`LossOfSignificance`) if the residual test
    /// rejected the solution.
    pub fn into_accepted(self) -> Result<Matrix<S>, SciOxideError> {
        self.report.termination.require_accepted()?;
        Ok(self.solution)
    }

    /// Returns the diagnostics of the solve.
    pub const fn report(&self) -> MatrixEquationReport<S::Real> {
        self.report
    }
}

/// Prepared Schur form `A = Q T Q^H` of one coefficient.
#[derive(Debug, Clone)]
struct SchurFactor<S: NumericScalar> {
    t: Matrix<S>,
    q: Matrix<S>,
}
impl<S: NumericScalar> SchurFactor<S> {
    // Stage a validated square view as the initial T.
    fn stage(view: MatrixView<'_, S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            t: view.try_to_matrix()?,
            q: Matrix::zeros(view.rows(), view.rows())?,
        })
    }
}

/// How the solution couples to the right coefficient.
#[derive(Debug, Clone)]
enum Coupling<S: NumericScalar> {
    /// `A X ± X B = C` with a separately factored B.
    Sylvester {
        b: Matrix<S>,
        right: SchurFactor<S>,
        sign: SylvesterSign,
    },
    /// Lyapunov equations reuse A's Schur form on both sides.
    Lyapunov(LyapunovKind),
}

#[derive(Debug, Clone)]
struct EquationPlan<S: NumericScalar> {
    a: Matrix<S>,
    left: SchurFactor<S>,
    coupling: Coupling<S>,
    options: MatrixEquationOptions<S::Real>,
}

/// Sylvester equation with fixed coefficients, prepared for repeated solves.
///
/// The plan solves
///
/// ```text
/// A X + s X B = C,
/// ```
///
/// where `A` is `m × m`, `B` is `n × n`, the right-hand side `C` and the
/// solution `X` are `m × n`, and `s = ±1` is given by [`SylvesterSign`]. The
/// solution is unique if and only if no eigenvalue `λ` of `A` and `μ` of `B`
/// satisfy `λ + s μ = 0`. For other orientations, such as `Aᵀ X + X B = C`,
/// pass a transposed or adjoint view of the coefficient.
///
/// The plan uses the Bartels–Stewart method. Construction computes the Schur
/// forms `A = Q_A T_A Q_Aᴴ` and `B = Q_B T_B Q_Bᴴ` once in O(m³ + n³) time. Real
/// input uses the real Schur form with 1×1 and 2×2 diagonal blocks. Each solve
/// forms `Q_Aᴴ C Q_B`, solves the quasi-triangular equation by block
/// substitution, transforms back, and checks the residual of the original
/// equation as described on [`MatrixEquationOptions`]. A solve costs
/// O(m²n + mn²) operations.
///
/// ```
/// use scioxide_linalg::{Matrix, MatrixEquationOptions, SylvesterPlan, SylvesterSign};
/// let a = Matrix::from_vec(2, 2, vec![3.0_f64, 1.0, 0.0, 2.0])?;
/// let b = Matrix::from_vec(1, 1, vec![1.0_f64])?;
/// let plan = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, MatrixEquationOptions::new())?;
/// let c = Matrix::from_vec(2, 1, vec![5.0_f64, 3.0])?;
/// let result = plan.solve(&c)?;
/// // (A + I) x = c is [[4, 1], [0, 3]] x = [5, 3], so x = [1, 1].
/// assert!((result.solution().unwrap().view().get(1, 0)? - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. H. Bartels and G. W. Stewart, "Algorithm 432: Solution of the matrix
///   equation AX + XB = C", *Communications of the ACM* 15(9), 1972.
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §7.6.
#[derive(Debug, Clone)]
pub struct SylvesterPlan<S: NumericScalar>(EquationPlan<S>);

/// Lyapunov equation with a fixed coefficient, prepared for repeated solves.
///
/// For an `n × n` matrix `A` and right-hand side `Q`, the plan solves one of
///
/// ```text
/// A X + X Aᴴ = Q          (LyapunovKind::Continuous)
/// A X Aᴴ − X + Q = 0      (LyapunovKind::Discrete)
/// ```
///
/// for the `n × n` matrix `X`. The continuous equation has a unique solution if
/// and only if `λᵢ + conj(λⱼ) ≠ 0` for all eigenvalues `λᵢ`, `λⱼ` of `A`. The
/// discrete equation has one if and only if `λᵢ conj(λⱼ) ≠ 1`. For
/// `Aᴴ X + X A = Q`, pass the adjoint view of `A`.
///
/// The method is the Bartels–Stewart method of [`SylvesterPlan`], with one
/// Schur form `A = Q_A T_A Q_Aᴴ` used on both sides. Construction computes
/// that Schur form once in O(n³) operations; each solve costs O(n³). When `Q` is exactly
/// Hermitian, the computed solution is replaced by its Hermitian part (see
/// [`MatrixEquationReport::symmetry_defect`]).
///
/// ```
/// use scioxide_linalg::{LyapunovKind, LyapunovPlan, Matrix, MatrixEquationOptions};
/// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
/// let plan = LyapunovPlan::new(&a, LyapunovKind::Continuous, MatrixEquationOptions::new())?;
/// let q = Matrix::from_vec(1, 1, vec![-2.0_f64])?;
/// // −x − x = −2 gives x = 1.
/// assert!((plan.solve(&q)?.solution().unwrap().view().get(0, 0)? - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. H. Bartels and G. W. Stewart, "Algorithm 432: Solution of the matrix
///   equation AX + XB = C", *Communications of the ACM* 15(9), 1972.
#[derive(Debug, Clone)]
pub struct LyapunovPlan<S: NumericScalar>(EquationPlan<S>);

impl<S: NumericScalar> SylvesterPlan<S> {
    /// Computes the Schur forms of `A` and `B` and returns the plan.
    ///
    /// This is the owned form of [`Self::build`] (see [execution](crate#execution)).
    ///
    /// # Errors
    /// - The errors of [`Self::build`].
    /// - `CapabilityUnavailable` if the options set a time limit.
    pub fn new<'a, 'b>(
        a: impl Into<MatrixView<'a, S>>,
        b: impl Into<MatrixView<'b, S>>,
        sign: SylvesterSign,
        options: MatrixEquationOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(a, b, sign, options, ExecutionContext::sequential_native())
    }

    /// Computes the Schur forms of `A` and `B` under `context` and returns the plan.
    ///
    /// The QR iterations of the Schur reduction count as nested work of
    /// `context`. The plan does not keep `context`.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_linalg::{Matrix, MatrixEquationOptions, SylvesterPlan, SylvesterSign};
    /// let a = Matrix::<f64>::identity(2)?;
    /// let plan = SylvesterPlan::build(
    ///     &a,
    ///     &a,
    ///     SylvesterSign::Plus,
    ///     MatrixEquationOptions::new(),
    ///     ExecutionContext::sequential_native(),
    /// )?;
    /// // A = B = I, so 2 X = I.
    /// assert_eq!(plan.solve(&a)?.solution().unwrap().as_slice(), &[0.5, 0.0, 0.0, 0.5]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    /// # Errors
    /// - `DimensionMismatch` if `A` or `B` is not square.
    /// - `NonFinite` if `A` or `B` has a NaN or infinite entry.
    /// - `ConvergenceFailure` if the QR iteration of a Schur reduction does not
    ///   converge.
    /// - `ResourceLimit` or `AllocationFailure` for the reduction's scratch storage.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn build<'a, 'b, 'scope, 'capability>(
        a: impl Into<MatrixView<'a, S>>,
        b: impl Into<MatrixView<'b, S>>,
        sign: SylvesterSign,
        options: MatrixEquationOptions<S::Real>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let scope = context.into().with_options(&options.execution_options())?;
        scope.kernel().execution().check()?;
        let (a, b) = (a.into(), b.into());
        crate::checked::validate_matrix(a, a.rows(), a.rows())?;
        crate::checked::validate_matrix(b, b.rows(), b.rows())?;
        let mut left = SchurFactor::stage(a)?;
        let mut right = SchurFactor::stage(b)?;
        factor(&mut [&mut left, &mut right], &options, scope)?;
        Ok(Self(EquationPlan {
            a: a.try_to_matrix()?,
            left,
            coupling: Coupling::Sylvester {
                b: b.try_to_matrix()?,
                right,
                sign,
            },
            options,
        }))
    }
}

impl<S: NumericScalar> LyapunovPlan<S> {
    /// Computes the Schur form of `A` and returns the plan.
    ///
    /// This is the owned form of [`Self::build`] (see [execution](crate#execution)).
    ///
    /// # Errors
    /// - The errors of [`Self::build`].
    /// - `CapabilityUnavailable` if the options set a time limit.
    pub fn new<'a>(
        a: impl Into<MatrixView<'a, S>>,
        kind: LyapunovKind,
        options: MatrixEquationOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(a, kind, options, ExecutionContext::sequential_native())
    }

    /// Computes the Schur form of `A` under `context` and returns the plan.
    ///
    /// The QR iterations of the Schur reduction count as nested work of
    /// `context`. The plan does not keep `context`.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_linalg::{LyapunovKind, LyapunovPlan, Matrix, MatrixEquationOptions};
    /// let a = Matrix::from_vec(1, 1, vec![0.5_f64])?;
    /// let plan = LyapunovPlan::build(
    ///     &a,
    ///     LyapunovKind::Discrete,
    ///     MatrixEquationOptions::new(),
    ///     ExecutionContext::sequential_native(),
    /// )?;
    /// let q = Matrix::from_vec(1, 1, vec![0.75_f64])?;
    /// // 0.25 x − x + 0.75 = 0 gives x = 1.
    /// assert!((plan.solve(&q)?.solution().unwrap().view().get(0, 0)? - 1.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    /// # Errors
    /// - `DimensionMismatch` if `A` is not square.
    /// - `NonFinite` if `A` has a NaN or infinite entry.
    /// - `ConvergenceFailure` if the QR iteration of the Schur reduction does not
    ///   converge.
    /// - `ResourceLimit` or `AllocationFailure` for the reduction's scratch storage.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn build<'a, 'scope, 'capability>(
        a: impl Into<MatrixView<'a, S>>,
        kind: LyapunovKind,
        options: MatrixEquationOptions<S::Real>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let scope = context.into().with_options(&options.execution_options())?;
        scope.kernel().execution().check()?;
        let a = a.into();
        crate::checked::validate_matrix(a, a.rows(), a.rows())?;
        let mut left = SchurFactor::stage(a)?;
        factor(&mut [&mut left], &options, scope)?;
        Ok(Self(EquationPlan {
            a: a.try_to_matrix()?,
            left,
            coupling: Coupling::Lyapunov(kind),
            options,
        }))
    }
}

macro_rules! solve_forms {
    ($name:ident, $example:literal) => {
        impl<S: NumericScalar> $name<S> {
            /// Returns the workspace size needed by [`Self::solve_into`].
            ///
            /// The workspace holds about five solution-sized matrices, four
            /// without the separation estimate, plus fixed tile scratch.
            ///
            /// # Errors
            /// - `SizeOverflow` if the size does not fit in `usize`.
            /// - `ResourceLimit` if it exceeds the configured byte limit.
            pub fn workspace_requirements(
                &self,
                (): (),
            ) -> Result<LinalgRequirements, SciOxideError> {
                self.0.requirements()
            }

            /// Allocates a workspace for [`Self::solve_into`].
            ///
            /// # Errors
            /// - The errors of [`Self::workspace_requirements`].
            /// - `AllocationFailure` if the allocation fails.
            pub fn workspace(&self, (): ()) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
                let mut workspace = LinalgWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(())?)?;
                Ok(workspace)
            }

            /// Solves the equation for one right-hand side.
            ///
            /// This is the owned form of [`Self::solve_into`]
            /// (see [execution](crate#execution)).
            ///
            /// # Errors
            /// - The errors of [`Self::solve_into`].
            /// - `CapabilityUnavailable` if the options set a time limit.
            pub fn solve<'r>(
                &self,
                rhs: impl Into<MatrixView<'r, S>>,
            ) -> Result<MatrixEquationResult<S>, SciOxideError> {
                let (rows, cols) = self.0.shape();
                let mut solution = Matrix::zeros(rows, cols)?;
                let report = self.solve_into(
                    rhs,
                    solution.view_mut(),
                    &mut LinalgWorkspace::new(),
                    ExecutionContext::sequential_native(),
                )?;
                Ok(MatrixEquationResult { solution, report })
            }

            /// Solves the equation for the right-hand side `rhs` and writes the
            /// solution to `output`.
            ///
            /// `rhs` and `output` have the shape of the solution. If the residual
            /// test fails, the call returns `Ok` with
            /// [`DirectSolveTermination::ResidualRejected`] and does not write
            /// `output` (see [direct solves](crate#direct-solves)). Work units
            /// count as nested work of `context`.
            ///
            #[doc = $example]
            /// # Errors
            /// - `DimensionMismatch` if `rhs` or `output` does not have the
            ///   solution shape.
            /// - `NonFinite` if `rhs` has a NaN or infinite entry.
            /// - `Singular` if a diagonal-block system of the Schur-form
            ///   equation has an exactly zero pivot, which happens when the
            ///   eigenvalue condition for a unique solution fails in floating point.
            /// - `ConvergenceFailure` if the solve needs more work units than the
            ///   limit set on [`MatrixEquationOptions`].
            /// - `NumericalInstability` if the solution or a reported diagnostic
            ///   overflows the scalar type.
            /// - `ExecutionStopped` on cancellation or deadline.
            /// - Workspace errors as described in
            ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
            pub fn solve_into<'r, 'scope, 'capability>(
                &self,
                rhs: impl Into<MatrixView<'r, S>>,
                output: MatrixViewMut<'_, S>,
                workspace: &mut LinalgWorkspace<'_, S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<MatrixEquationReport<S::Real>, SciOxideError> {
                self.0.run(rhs.into(), output, workspace, context.into())
            }
        }
    };
}
solve_forms!(
    SylvesterPlan,
    "```
use scioxide_core::ExecutionContext;
use scioxide_linalg::{Matrix, MatrixEquationOptions, SylvesterPlan, SylvesterSign};
let a = Matrix::<f64>::identity(2)?;
let plan = SylvesterPlan::new(&a, &a, SylvesterSign::Plus, MatrixEquationOptions::new())?;
let mut workspace = plan.workspace(())?;
let mut x = Matrix::zeros(2, 2)?;
let report = plan.solve_into(
    &a,
    x.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
// A = B = I, so 2 X = I.
assert!(report.residual() <= 1e-15);
assert_eq!(x.as_slice(), &[0.5, 0.0, 0.0, 0.5]);
# Ok::<(), scioxide_core::SciOxideError>(())
```"
);
solve_forms!(
    LyapunovPlan,
    "```
use scioxide_core::ExecutionContext;
use scioxide_linalg::{
    LyapunovKind, LyapunovPlan, Matrix, MatrixEquationOptions, DirectSolveTermination,
};
let a = Matrix::from_vec(2, 2, vec![-1.0_f64, 0.25, 0.0, -2.0])?;
let plan = LyapunovPlan::new(&a, LyapunovKind::Continuous, MatrixEquationOptions::new())?;
let mut workspace = plan.workspace(())?;
let mut x = Matrix::zeros(2, 2)?;
let q = Matrix::<f64>::identity(2)?.view().try_scale(-1.0)?;
let report = plan.solve_into(
    &q,
    x.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), DirectSolveTermination::Accepted);
# Ok::<(), scioxide_core::SciOxideError>(())
```"
);

/// Factor every staged coefficient in place with the arena Schur kernel, in a
/// child epoch of `context`. Preparation scratch is local to the call.
fn factor<S: NumericScalar>(
    factors: &mut [&mut SchurFactor<S>],
    options: &MatrixEquationOptions<S::Real>,
    context: LinalgExecutionContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let order = factors.iter().map(|f| f.t.rows()).max().unwrap_or(0);
    let lengths = crate::eigen::schur_kernel_lengths::<S>(
        order,
        options.execution_options().execution_policy(),
    )?;
    let mut scratch = LinalgRequirements::new(options.execution_options().byte_limit());
    *scratch.scalar_mut(ScalarRegion::Direct) = lengths.scalars;
    *scratch.real_mut(RealRegion::Direct) = lengths.reals;
    let mut workspace = LinalgWorkspace::new();
    workspace.try_reserve(&scratch)?;
    let schur = SchurOptions::new().with_execution_options(options.execution_options());
    workspace.run_scope(
        context,
        identity(order, order),
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, _, solve| {
            workspace.with_direct(|scalars, reals, _, _| {
                for factor in factors.iter_mut() {
                    let n = factor.t.rows();
                    crate::eigen::schur_kernel(
                        factor.t.as_mut_slice(),
                        factor.q.as_mut_slice(),
                        n,
                        &mut *scalars,
                        &mut *reals,
                        &schur,
                        solve,
                    )?;
                    crate::eigen::for_each_schur_block(factor.t.as_slice(), n, |_, _| {})?;
                }
                solve.execution().check()
            })
        },
    )
}

fn identity(rows: usize, cols: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "matrix_equation", 1),
        ((rows as u128) << 64) | cols as u128,
        0,
    )
}

/// Solve-time staging carved from the shared workspace.
struct Staging<'a, S: NumericScalar> {
    candidate: ScratchMatrix<'a, S>,
    x: Framed<'a, S>,
    scratch: Framed<'a, S>,
    product: Framed<'a, S>,
    estimate: &'a mut [ScaledValue<S>],
    blocks: &'a mut [ScaledValue<S>],
    block_values: &'a mut [S],
}

impl<S: NumericScalar> EquationPlan<S> {
    // Solution rows and columns.
    fn shape(&self) -> (usize, usize) {
        let rows = self.a.rows();
        match &self.coupling {
            Coupling::Sylvester { b, .. } => (rows, b.rows()),
            Coupling::Lyapunov(_) => (rows, rows),
        }
    }
    // Right coefficient: B, or A^H for Lyapunov equations.
    fn b(&self) -> MatrixView<'_, S> {
        match &self.coupling {
            Coupling::Sylvester { b, .. } => b.view(),
            Coupling::Lyapunov(_) => self.a.view().adjoint(),
        }
    }
    fn right(&self) -> &SchurFactor<S> {
        match &self.coupling {
            Coupling::Sylvester { right, .. } => right,
            Coupling::Lyapunov(_) => &self.left,
        }
    }
    fn sign(&self) -> S {
        match self.coupling {
            Coupling::Sylvester {
                sign: SylvesterSign::Plus,
                ..
            }
            | Coupling::Lyapunov(LyapunovKind::Continuous) => S::ONE,
            Coupling::Sylvester {
                sign: SylvesterSign::Minus,
                ..
            }
            | Coupling::Lyapunov(LyapunovKind::Discrete) => -S::ONE,
        }
    }
    fn discrete(&self) -> bool {
        matches!(self.coupling, Coupling::Lyapunov(LyapunovKind::Discrete))
    }
    fn lyapunov(&self) -> bool {
        matches!(self.coupling, Coupling::Lyapunov(_))
    }

    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        let (rows, cols) = self.shape();
        requirements::<S>(rows, cols, self.options.separation, &self.options.execution)
    }

    fn run(
        &self,
        rhs: MatrixView<'_, S>,
        mut output: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<MatrixEquationReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.with_options(&self.options.execution_options())?;
        scope.kernel().execution().check()?;
        let (m, n) = self.shape();
        crate::checked::validate_matrix(rhs, m, n)?;
        if (output.rows(), output.cols()) != (m, n) {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(m, n),
                Dimensions::two(output.rows(), output.cols()),
            ));
        }
        let requirements = self
            .requirements()?
            .max(&blocked::packing_requirements::<S>(
                &self.options.execution,
                Some(scope.kernel()),
            )?);
        workspace.try_reserve(&requirements)?;
        let elements = m * n;
        let estimate = match self.options.separation {
            Estimation::Skip => 0,
            Estimation::Estimate => elements,
        };
        workspace.run_scope(
            scope,
            identity(m, n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                workspace.with_direct(|mut scalars, _, indices, child| {
                    let (row_cuts, col_cuts) =
                        crate::workspace::split_mut(indices, m.div_ceil(blocked::TILE) + 1)?;
                    let col_cuts = &mut col_cuts[..n.div_ceil(blocked::TILE) + 1];
                    blocked::partition(self.left.t.view(), row_cuts);
                    blocked::partition(self.right().t.view(), col_cuts);
                    let grid = Grid {
                        rows: row_cuts,
                        cols: col_cuts,
                    };
                    child.with_scaled(|mut scaled, mut exponents, gemm| {
                        let mut staging = Staging {
                            candidate: ScratchMatrix::new(take_mut(&mut scalars, elements)?, m, n)?,
                            x: Framed::new(
                                take_mut(&mut scalars, elements)?,
                                take_mut(&mut scaled, elements)?,
                                take_mut(&mut exponents, grid.blocks())?,
                                grid,
                            ),
                            scratch: Framed::new(
                                take_mut(&mut scalars, elements)?,
                                take_mut(&mut scaled, elements)?,
                                take_mut(&mut exponents, grid.blocks())?,
                                grid,
                            ),
                            product: Framed::new(
                                take_mut(&mut scalars, elements)?,
                                take_mut(&mut scaled, elements)?,
                                take_mut(&mut exponents, grid.blocks())?,
                                grid,
                            ),
                            estimate: take_mut(&mut scaled, estimate)?,
                            blocks: take_mut(&mut scaled, blocked::SCRATCH)?,
                            block_values: take_mut(&mut scalars, blocked::SCRATCH)?,
                        };
                        let mut products = Products {
                            packing: take_mut(&mut scalars, blocked::PACKING)?,
                            workspace: gemm,
                            kernels,
                            policy: ExecutionPolicy::Sequential,
                        };
                        let report =
                            self.solve_candidate(rhs, &mut staging, &mut products, solve)?;
                        solve.execution().check()?;
                        if report.termination.is_success() {
                            for i in 0..m {
                                for j in 0..n {
                                    output.set_element(i, j, staging.candidate.get_element(i, j));
                                }
                            }
                        }
                        Ok(report)
                    })
                })
            },
        )
    }

    fn solve_candidate(
        &self,
        rhs: MatrixView<'_, S>,
        w: &mut Staging<'_, S>,
        products: &mut Products<'_, '_, '_, S>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<MatrixEquationReport<S::Real>, SciOxideError> {
        let (m, n) = self.shape();
        let limit = self.options.work_limit().get();
        let mut work = 0;
        let mut admit = || {
            solve.execution().check()?;
            if work == limit || !solve.record_iteration()?.is_granted() {
                return Err(exhausted(solve.counts()));
            }
            work += 1;
            Ok(())
        };
        self.solve_schur_coordinates(rhs, w, products, &mut admit)?;
        let right = self.right();
        let lyapunov = self.lyapunov();
        let discrete = self.discrete();
        let sign = self.sign();
        // For a Hermitian right-hand side the Lyapunov solution is Hermitian
        // in exact arithmetic: report the computed defect and project onto
        // Hermitian matrices by averaging. Non-Hermitian data is left as is.
        let symmetry_defect = if lyapunov {
            let mut defect = S::Real::ZERO;
            let hermitian = (0..m)
                .all(|i| (0..n).all(|j| rhs.get_element(i, j) == rhs.get_element(j, i).conj()));
            for i in 0..m {
                for j in i..n {
                    let u = ScaledValue::from_value(w.candidate.get_element(i, j))?;
                    let v = ScaledValue::from_value(w.candidate.get_element(j, i).conj())?;
                    let d = u.sub(v)?.magnitude()?.round()?;
                    defect = defect.max(d);
                    if hermitian {
                        let average = u
                            .add(v)?
                            .mul(ScaledValue::from_value(S::from_real(S::Real::HALF))?)?
                            .round()?;
                        w.candidate.as_mut_slice()[i * n + j] = average;
                        w.candidate.as_mut_slice()[j * n + i] = average.conj();
                    }
                }
            }
            Some(defect)
        } else {
            None
        };
        let separation = if self.options.separation == Estimation::Estimate && m * n > 0 {
            let estimate = crate::estimate::separation_estimate(w.estimate, |x, adjoint| {
                w.x.wide.copy_from_slice(x);
                w.x.scales.fill(WIDE);
                let solved = blocked::solve(
                    self.left.t.view(),
                    right.t.view(),
                    sign,
                    &mut w.x,
                    &mut w.product,
                    w.block_values,
                    w.blocks,
                    adjoint,
                    adjoint ^ lyapunov,
                    discrete,
                    products,
                    &mut admit,
                )?;
                let input = w.x.input()?;
                for (k, value) in x.iter_mut().enumerate() {
                    *value = input.get(k / n, k % n)?;
                }
                Ok(solved)
            })?;
            Some(estimate.map_or(Ok(S::Real::ZERO), |s| s.round())?)
        } else {
            None
        };
        let (residual, scale) = residual(
            self.a.view(),
            self.b(),
            w.candidate.try_view()?,
            rhs,
            sign,
            discrete,
            &mut w.product,
            &mut w.scratch,
            products,
            &mut admit,
        )?;
        let tolerance = self.options.tolerance;
        let threshold = Scaled::from_value(tolerance.absolute())?
            .add(Scaled::from_value(tolerance.relative())?.mul(scale)?)?;
        let termination = if residual.abs_le(threshold) {
            DirectSolveTermination::Accepted
        } else {
            DirectSolveTermination::ResidualRejected
        };
        let backward_error = if scale.is_zero() {
            S::Real::ZERO
        } else {
            residual.div(scale)?.round()?
        };
        Ok(MatrixEquationReport {
            termination,
            residual: residual.round()?,
            backward_error,
            symmetry_defect,
            separation,
            work,
        })
    }

    fn solve_schur_coordinates(
        &self,
        rhs: MatrixView<'_, S>,
        w: &mut Staging<'_, S>,
        products: &mut Products<'_, '_, '_, S>,
        admit: &mut impl FnMut() -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        // Bartels-Stewart: with Schur forms `A = Q_a T_a Q_a^H` and
        // `B = Q_b T_b Q_b^H` prepared once, the equation becomes a
        // (quasi-)triangular one for `Y = Q_a^H X Q_b` with right-hand side
        // `Q_a^H C Q_b`, solved by block substitution, then `X = Q_a Y Q_b^H`.
        // Lyapunov equations reuse A's Schur form on both sides (with the
        // adjoint orientation for `X A^H`).
        let left = &self.left;
        let right = self.right();
        let discrete = self.discrete();
        let qr = &right.q;
        products.product(
            Input::Plain(left.q.view().adjoint()),
            Input::Plain(rhs),
            &mut w.scratch,
            admit,
        )?;
        products.product(w.scratch.input()?, Input::Plain(qr.view()), &mut w.x, admit)?;
        if discrete {
            w.x.negate();
        }
        let solved = blocked::solve(
            left.t.view(),
            right.t.view(),
            self.sign(),
            &mut w.x,
            &mut w.product,
            w.block_values,
            w.blocks,
            false,
            self.lyapunov(),
            discrete,
            products,
            admit,
        )?;
        if !solved {
            return Err(SciOxideError::singular(None));
        }
        products.product(
            Input::Plain(left.q.view()),
            w.x.input()?,
            &mut w.scratch,
            admit,
        )?;
        products.product(
            w.scratch.input()?,
            Input::Plain(qr.view().adjoint()),
            &mut w.x,
            admit,
        )?;
        w.x.publish(w.candidate.as_mut_slice())?;
        Ok(())
    }
}

/// Shared-arena requirements of an `rows x cols` solve.
fn requirements<S: NumericScalar>(
    rows: usize,
    cols: usize,
    separation: Estimation,
    execution: &LinalgExecutionOptions,
) -> Result<LinalgRequirements, SciOxideError> {
    let elements = checked_mul(rows, cols, scioxide_core::quantity!("equation_elements"))?;
    let buffers = match separation {
        Estimation::Skip => 3,
        Estimation::Estimate => 4,
    };
    let mut requirements = blocked::packing_requirements::<S>(execution, None)?;
    let size = scioxide_core::quantity!("equation_scratch");
    *requirements.scalar_mut(ScalarRegion::Direct) = checked_add(
        checked_mul(4, elements, size)?,
        blocked::PACKING + blocked::SCRATCH,
        size,
    )?;
    requirements.scaled = checked_add(
        checked_mul(elements, buffers, size)?,
        blocked::SCRATCH,
        size,
    )?;
    let row_blocks = rows.div_ceil(blocked::TILE);
    let col_blocks = cols.div_ceil(blocked::TILE);
    requirements.exponents = checked_mul(3, checked_mul(row_blocks, col_blocks, size)?, size)?;
    requirements.indices = checked_add(checked_add(row_blocks, col_blocks, size)?, 2, size)?;
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

fn exhausted(counts: scioxide_core::SolveCounts) -> SciOxideError {
    SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts),
        "matrix equation work budget exhausted",
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
#[expect(
    clippy::type_complexity,
    reason = "the tuple groups the distinct numerical outputs of this operation"
)]
fn residual<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    x: MatrixView<'_, S>,
    rhs: MatrixView<'_, S>,
    sign: S,
    discrete: bool,
    product: &mut Framed<'_, S>,
    scratch: &mut Framed<'_, S>,
    products: &mut Products<'_, '_, '_, S>,
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(Scaled<S::Real>, Scaled<S::Real>), SciOxideError> {
    products.product(Input::Plain(a), Input::Plain(x), product, &mut admit)?;
    if discrete {
        products.product(product.input()?, Input::Plain(b), scratch, &mut admit)?;
    } else {
        products.product(Input::Plain(x), Input::Plain(b), scratch, &mut admit)?;
    }
    let mut error = Scaled::ZERO;
    let mut norm = [Scaled::ZERO; 3];
    let grid = product.grid;
    for r in 0..grid.rows.len() - 1 {
        for c in 0..grid.cols.len() - 1 {
            admit()?;
            let (row, col) = (grid.rows[r], grid.cols[c]);
            let (rows, cols) = (grid.rows[r + 1] - row, grid.cols[c + 1] - col);
            let first = if discrete {
                scratch.input()?
            } else {
                product.input()?
            }
            .block(row, col, rows, cols)?;
            let second = if discrete {
                Input::Plain(x)
            } else {
                scratch.input()?
            }
            .block(row, col, rows, cols)?;
            let third = Input::Plain(rhs.block(row, col, rows, cols)?);
            let (e, terms) = blocked::residual_tile(
                [first, second, third],
                [
                    S::ONE,
                    if discrete { -S::ONE } else { sign },
                    if discrete { S::ONE } else { -S::ONE },
                ],
                products.packing,
            )?;
            error = error.add(e)?;
            for (sum, value) in norm.iter_mut().zip(terms) {
                *sum = sum.add(value)?;
            }
        }
    }
    let scale = norm
        .into_iter()
        .try_fold(Scaled::ZERO, |acc, v| acc.add(v.sqrt()?))?;
    Ok((error.sqrt()?, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_size_only_requested_estimates_and_overflow() {
        let skip =
            requirements::<f64>(3, 2, Estimation::Skip, &LinalgExecutionOptions::new()).unwrap();
        let estimate =
            requirements::<f64>(3, 2, Estimation::Estimate, &LinalgExecutionOptions::new())
                .unwrap();
        assert_eq!(
            skip.scalar(ScalarRegion::Direct),
            24 + blocked::PACKING + blocked::SCRATCH
        );
        assert_eq!(skip.scaled, 18 + blocked::SCRATCH);
        assert_eq!(estimate.scaled, 24 + blocked::SCRATCH);
        for (rows, cols) in [(usize::MAX, 2), (usize::MAX / 2, 1)] {
            assert_eq!(
                requirements::<f64>(
                    rows,
                    cols,
                    Estimation::Estimate,
                    &LinalgExecutionOptions::new()
                )
                .unwrap_err()
                .kind(),
                scioxide_core::SciOxideErrorKind::SizeOverflow
            );
        }
    }
}
