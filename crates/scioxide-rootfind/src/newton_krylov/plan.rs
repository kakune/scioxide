//! Newton–Krylov solver value and its owned and prepared solves.
use super::*;

/// Solves `F(x) = 0` for `F: Tⁿ → Tⁿ` by the inexact Newton–Krylov method, without
/// forming the Jacobian.
///
/// Each Newton step solves `J(x) d = −F(x)` approximately by restarted GMRES
/// ([`GmresSolver`]) to the relative tolerance `η` of the
/// [`NewtonKrylovForcing`], then takes `x + α d` with a nonmonotone backtracking
/// line search on `½‖F‖²`. GMRES needs only Jacobian-vector products `J v`. They
/// come from a [`JacobianOperator`](scioxide_core::JacobianOperator), or from one
/// forward difference of `F` per product (see [`NewtonKrylovProblem`]). An
/// optional right preconditioner may change between applications. Each GMRES
/// iteration costs one product and `O(n m)` work for restart length `m`, and the
/// workspace holds `O(n m)` values. For small dense systems, a solver that
/// factors the Jacobian, such as [`NewtonSystem`](crate::NewtonSystem), is usually
/// faster.
///
/// After each inner solve the solver recomputes `‖F(x) + J d‖` itself. An inner
/// solve that missed `η` is still used if this residual stays within the largest
/// forcing term, because `d` is then a descent direction for `½‖F‖²`. Otherwise
/// the solve stops with [`RootTermination::LinearFailure`]. Other unsuccessful
/// statuses are [`RootTermination::GlobalizationFailure`] (the line search found no
/// acceptable step), [`RootTermination::Stagnation`],
/// [`RootTermination::IterationLimit`] and the budget and execution stops listed in
/// [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver, ResidualFn};
///
/// // F(x) = x² − 2 has the root √2.
/// let residual = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] - 2.0;
///     Ok(())
/// });
/// let problem = NewtonKrylovProblem::numerical(&residual);
/// let solver = NewtonKrylovSolver::new(NewtonKrylovOptions::default())?;
/// let result = solver.solve(&problem, &[1.0])?;
/// assert!(result.is_success());
/// assert!((result.point()[0] - 2.0_f64.sqrt()).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. S. Dembo, S. C. Eisenstat and T. Steihaug, "Inexact Newton methods",
///   *SIAM J. Numer. Anal.* 19(2), 1982.
/// - C. T. Kelley, *Solving Nonlinear Equations with Newton's Method*, SIAM, 2003.
/// - D. A. Knoll and D. E. Keyes, "Jacobian-free Newton–Krylov methods: a survey
///   of approaches and applications", *J. Comput. Phys.* 193(2), 2004.
#[derive(Debug, Clone)]
pub struct NewtonKrylovSolver<T: NumericScalar> {
    pub(crate) options: NewtonKrylovOptions<T::Real>,
    pub(crate) scalar: core::marker::PhantomData<T>,
}
impl<T: NumericScalar> NewtonKrylovSolver<T> {
    /// Creates a solver after validating `options`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if the forcing bounds are not in `(0, 1)` or not ordered, the
    /// residual scale is not positive and finite, the Armijo constant is not in
    /// `(0, 1)`, the smallest relative step is not in `(0, 1]`, or the nonmonotone
    /// weight is not in `[0, 1]`.
    pub fn new(options: NewtonKrylovOptions<T::Real>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            options,
            scalar: core::marker::PhantomData,
        })
    }
    /// Returns the validated options.
    pub const fn options(&self) -> &NewtonKrylovOptions<T::Real> {
        &self.options
    }
    /// Returns the workspace size needed for `problem`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `problem` is empty or not square, if its preconditioner
    ///   has another dimension, or if it has both a preconditioner and a linear
    ///   solver.
    /// - `SizeOverflow` if the size does not fit in `usize`, and `ResourceLimit` if
    ///   it exceeds [`NewtonKrylovOptions::with_workspace_byte_limit`].
    /// - Errors from the scratch-size queries of the residual operator, the
    ///   Jacobian operator or the scaling.
    pub fn workspace_requirements(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
    ) -> Result<NewtonKrylovWorkspaceRequirements<T>, SciOxideError> {
        NewtonKrylovWorkspaceRequirements::new(problem, &self.options)
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure` if
    /// memory cannot be reserved.
    pub fn workspace(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
    ) -> Result<NewtonKrylovWorkspace<T>, SciOxideError> {
        NewtonKrylovWorkspace::allocate(&self.workspace_requirements(problem)?)
    }
    /// Solves from `initial` and returns a new result.
    ///
    /// Allocates a workspace and a result, and runs sequentially with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    /// Use [`Self::solve_into`] to reuse storage.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::solve_into`] and [`Self::workspace`].
    pub fn solve<'a, 'layout>(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
        initial: impl Into<NewtonKrylovInput<'a, 'layout, T>>,
    ) -> Result<NewtonKrylovResult<T>, SciOxideError>
    where
        T: 'a,
    {
        let mut workspace = self.workspace(problem)?;
        let mut output = NewtonKrylovResult::new(problem.dimension(), &self.options)?;
        self.solve_into(
            problem,
            initial,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Input checks shared by standalone solves and solves inside another solver.
    pub(crate) fn prepare(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
        initial: NewtonKrylovInput<'_, '_, T>,
        output: &NewtonKrylovResult<T>,
        w: &mut NewtonKrylovWorkspace<T>,
    ) -> Result<(), SciOxideError> {
        let requirements = self.workspace_requirements(problem)?;
        if initial.len()? != problem.dimension()
            || !output.has_capacity(problem.dimension(), &self.options)
        {
            return Err(invalid(
                scioxide_core::argument!("initial"),
                InvalidInputReason::Inconsistent,
                "initial point and result must match problem dimension and history",
            ));
        }
        if w.check(&requirements)? != scioxide_core::Capacity::Sufficient {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                InvalidInputReason::Inconsistent,
                "reserve matching structural requirements",
            ));
        }
        match initial {
            NewtonKrylovInput::Slice(values) => {
                finite(values, NonFiniteOrigin::Input)?;
                w.x.as_mut_slice().copy_from_slice(values);
            }
            NewtonKrylovInput::View(values) => {
                for (to, &from) in w.x.as_mut_slice().iter_mut().zip(values.iter()) {
                    *to = from;
                }
                finite(w.x.as_slice(), NonFiniteOrigin::Input)?;
            }
        }
        w.accepted = None;
        Ok(())
    }
    /// Solves from `initial` into `output`, using `workspace` for scratch storage.
    ///
    /// `initial` is a slice or a rank-one view ([`NewtonKrylovInput`]). `scope` is
    /// an [execution context](scioxide_core#execution-contexts) or the
    /// [`SolveContext`](scioxide_core::SolveContext) of an enclosing solve. When a
    /// budget, cancellation or deadline stops the solve after the initial point has
    /// been evaluated, `output` holds the last accepted point with an unsuccessful
    /// [`RootTermination`]. On error, `output` is unchanged
    /// (see [Output on error](scioxide_core#output-on-error)).
    ///
    /// # Errors
    ///
    /// - The `problem` errors of [`Self::workspace_requirements`].
    /// - `InvalidInput` if `initial` does not have the problem dimension or is a
    ///   view of rank other than one, if `output` was not created for this
    ///   dimension and history setting, or if `workspace` was not prepared for this
    ///   solver and problem.
    /// - `NonFinite` if `initial` is not finite, or if the residual at the initial
    ///   point or a Jacobian-vector product is not finite.
    /// - `ExecutionStopped` or `ConvergenceFailure` if a stop occurs before the
    ///   initial point is evaluated.
    /// - Errors returned by the residual operator, the Jacobian operator, the
    ///   preconditioner, the linear solver or the scaling, unchanged. A trial point
    ///   whose residual is not finite, or whose evaluation returns `DomainError`
    ///   with reason `OutsideDomain`, is rejected by the line search instead.
    pub fn solve_into<'a, 'layout, 'scope, 'capability>(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
        initial: impl Into<NewtonKrylovInput<'a, 'layout, T>>,
        output: &mut NewtonKrylovResult<T>,
        w: &mut NewtonKrylovWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        T: 'a,
    {
        let initial = initial.into();
        crate::attempt::attempt(
            w,
            output,
            scope.into(),
            scioxide_core::AttemptKind::Call,
            |w, output| {
                self.prepare(problem, initial, output, w)?;
                Ok(crate::attempt::Entry::call(
                    scioxide_core::SolveIdentity::new(
                        scioxide_core::ProblemToken::new(0),
                        scioxide_core::algorithm_id!("rootfind", "Newton-Krylov", 1),
                        problem.dimension() as u128,
                        0,
                    ),
                    crate::context::cap(
                        self.options.budgets,
                        BudgetKind::Iteration,
                        self.options.iterations,
                    ),
                ))
            },
            |w, output, context| self.run(problem, &self.options, output, w, context),
            |w, output, _| crate::attempt::Conclusion {
                succeeded: output.is_success(),
                progress: w.accepted,
            },
        )
    }
}
