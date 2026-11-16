//! Problem for Newton–Krylov solves.
use super::*;
/// Describes the Newton system `J d = −F(x)` passed to a [`NonlinearLinearSolver`].
///
/// With row divisors `R` and variable divisors `V` from [`NonlinearScaling`] (all
/// ones without scaling), [`operator`](Self::operator) applies the scaled Jacobian
/// `R⁻¹ J V`. The point and the residual are in the units of `F` and `x`. The
/// divisors stay fixed during the linear solve and the line search that follows.
///
/// A solver that assembles a matrix writes entry `(i, j)` as
/// `J[i, j] · V[j] / R[i]`; an iterative solver applies [`Self::operator`].
///
/// # Examples
///
/// The example solves the 1 × 1 Newton system exactly from `J = 2x`.
///
/// ```
/// # use scioxide_core::{SciOxideError, PartialResultTrust, ResidualTolerance, SolveContext};
/// # use scioxide_linalg::{IterativeSolveReport, LinearSolveTermination, Vector};
/// # use scioxide_rootfind::{
/// #     NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver, NonlinearLinearSolver,
/// #     NonlinearLinearization, ResidualFn,
/// # };
/// struct ScalarCorrection;
/// impl NonlinearLinearSolver<f64> for ScalarCorrection {
///     fn solve_into(
///         &self,
///         view: &NonlinearLinearization<'_, f64>,
///         rhs: &Vector<f64>,
///         out: &mut Vector<f64>,
///         _: ResidualTolerance<f64>,
///         _: &mut SolveContext<'_, '_>,
///     ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
///         // Entry (0, 0) of R⁻¹ J V for J = 2x.
///         let derivative =
///             2.0 * view.point()[0] * view.variable_divisors()[0] / view.row_divisors()[0];
///         out.as_mut_slice()[0] = rhs.as_slice()[0] / derivative;
///         Ok(IterativeSolveReport::new(
///             LinearSolveTermination::Converged,
///             0,
///             None,
///             PartialResultTrust::DiagnosticOnly,
///         ))
///     }
/// }
///
/// // F(x) = x² − 2 has the root √2.
/// let residual = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] - 2.0;
///     Ok(())
/// });
/// let problem = NewtonKrylovProblem::numerical(&residual).with_linear_solver(&ScalarCorrection);
/// let result =
///     NewtonKrylovSolver::new(NewtonKrylovOptions::default())?.solve(&problem, &[1.0])?;
/// assert!(result.is_success());
/// # Ok::<(), SciOxideError>(())
/// ```
pub struct NonlinearLinearization<'a, T: NumericScalar> {
    pub(crate) operator: &'a dyn LinearOperator<T>,
    pub(crate) point: &'a [T],
    pub(crate) residual: &'a [T],
    pub(crate) rows: &'a [T::Real],
    pub(crate) variables: &'a [T::Real],
}
impl<T: NumericScalar> NonlinearLinearization<'_, T> {
    /// Returns the scaled Jacobian operator `R⁻¹ J V`. It must be applied with
    /// the solve context passed to the linear solver.
    pub fn operator(&self) -> &dyn LinearOperator<T> {
        self.operator
    }
    /// Returns the current Newton iterate `x`.
    pub fn point(&self) -> &[T] {
        self.point
    }
    /// Returns `F(x)` at [`Self::point`], without the row divisors.
    pub fn residual(&self) -> &[T] {
        self.residual
    }
    /// Returns the row divisors `R`, all ones without scaling.
    pub fn row_divisors(&self) -> &[T::Real] {
        self.rows
    }
    /// Returns the variable divisors `V`, all ones without scaling.
    pub fn variable_divisors(&self) -> &[T::Real] {
        self.variables
    }
}

/// Solves the Newton linear systems of a [`NewtonKrylovSolver`] in place of GMRES.
///
/// Implement this trait to use a direct solver, or an iterative solver with its
/// own preconditioner, and attach it with
/// [`NewtonKrylovProblem::with_linear_solver`]. The Newton solver counts each call
/// as one linear solve; record nested work through the `context` argument. After
/// each call the Newton solver recomputes the linear residual, so a solution that
/// misses the tolerance is detected (see [`NewtonKrylovSolver`]). Report
/// nonconvergence through the returned [`IterativeSolveReport`].
pub trait NonlinearLinearSolver<T: NumericScalar> {
    /// Solves `R⁻¹ J V y = rhs` to the relative residual `tolerance`, starting from
    /// zero, and writes `y` to `output`.
    ///
    /// # Errors
    ///
    /// Return an error if the system cannot be solved, for example a
    /// `DimensionMismatch`, `Singular` or `AllocationFailure`. The Newton solve
    /// returns it unchanged.
    fn solve_into(
        &self,
        linearization: &NonlinearLinearization<'_, T>,
        rhs: &Vector<T>,
        output: &mut Vector<T>,
        tolerance: ResidualTolerance<T::Real>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<T::Real>, SciOxideError>;
}

/// Supplies the units of the equations and the unknowns of a Newton–Krylov
/// problem.
///
/// The solver divides residual component `i` by a row divisor `Rᵢ` and step
/// component `j` by a variable divisor `Vⱼ`. Use it when components of `F` or `x`
/// have very different magnitudes. The scaled norm `‖R⁻¹ F(x)‖₂` is used in the
/// stopping test, the line search and the forcing terms; the variable divisors
/// set the step test and the forward-difference step. The solver computes the
/// divisors at the initial point and again at every accepted point, and keeps
/// them fixed during each linear solve and line search. The residual and point
/// in the result stay in the units of `F` and `x`.
///
/// # Examples
///
/// The residual `F(x) = 10⁹ (x − 2)` is divided by `10⁹`.
///
/// ```
/// # use scioxide_core::{SciOxideError, SolveContext};
/// # use scioxide_rootfind::{
/// #     NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver, NonlinearScaling, ResidualFn,
/// # };
/// struct Units;
/// impl NonlinearScaling<f64> for Units {
///     fn scales_into(
///         &self,
///         _: &[f64],
///         rows: &mut [f64],
///         variables: &mut [f64],
///         _: &mut [f64],
///         _: &mut SolveContext<'_, '_>,
///     ) -> Result<(), SciOxideError> {
///         rows.fill(1e9);
///         variables.fill(1.0);
///         Ok(())
///     }
/// }
///
/// // The root of F(x) = 10⁹ (x − 2) is x = 2.
/// let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = 1e9 * (x[0] - 2.0);
///     Ok(())
/// });
/// let p = NewtonKrylovProblem::numerical(&f).with_scaling(&Units);
/// let result = NewtonKrylovSolver::new(NewtonKrylovOptions::default())?.solve(&p, &[0.0])?;
/// assert!(result.is_success());
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait NonlinearScaling<T: NumericScalar> {
    /// Returns the length of the `scratch` slice that
    /// [`scales_into`](Self::scales_into) needs (default 0).
    ///
    /// # Errors
    ///
    /// Return an error if the length cannot be computed, for example a
    /// `SizeOverflow`.
    fn scaling_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    /// Writes positive, finite row divisors to `residual` and variable divisors to
    /// `variables` for the point `point`. Both slices have the problem dimension.
    /// Record nested work through `context`.
    ///
    /// # Errors
    ///
    /// Return an error if the divisors cannot be computed. The Newton solve returns
    /// it unchanged. It reports a divisor that is not finite as `NonFinite`, and one
    /// that is zero or negative as `InvalidInput`.
    fn scales_into(
        &self,
        point: &[T],
        residual: &mut [T::Real],
        variables: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Describes the system `F(x) = 0` solved by a [`NewtonKrylovSolver`].
///
/// The problem borrows a square residual operator and, optionally, a
/// [`JacobianOperator`](scioxide_core::JacobianOperator) for exact Jacobian-vector
/// products, a right preconditioner, a [`NonlinearLinearSolver`] and a
/// [`NonlinearScaling`]. Without a Jacobian operator, products are approximated by
/// forward differences (see [`numerical`](Self::numerical)).
#[derive(Clone, Copy)]
pub struct NewtonKrylovProblem<'a, T: NumericScalar> {
    pub(crate) scaling: Option<&'a dyn NonlinearScaling<T>>,
    pub(crate) residual: &'a dyn ResidualOperator<T>,
    pub(crate) jacobian: Option<&'a dyn JacobianOperator<T>>,
    pub(crate) preconditioner: Option<&'a dyn Preconditioner<T>>,
    pub(crate) linear_solver: Option<&'a dyn NonlinearLinearSolver<T>>,
}
impl<'a, T: NumericScalar> NewtonKrylovProblem<'a, T> {
    /// Creates a problem whose Jacobian-vector products are forward differences.
    ///
    /// ```text
    /// J v ≈ (F(x + σ v) − F(x)) / σ,   σ = √ε / maxᵢ (|vᵢ| / max(1, |xᵢ|)),
    /// ```
    ///
    /// so the coordinate that moves most changes by `√ε · max(1, |xᵢ|)`. With
    /// [`NonlinearScaling`], the variable divisor replaces `max(1, |xᵢ|)`. Each
    /// product costs one residual evaluation, counted as
    /// [`EvaluationRole::Residual`], and has relative error of order `√ε`, so the
    /// forcing term is at least `√ε`. The probe point is formed with `1/σ`, so very
    /// small or very large directions do not overflow.
    pub fn numerical(residual: &'a dyn ResidualOperator<T>) -> Self {
        Self {
            scaling: None,
            residual,
            jacobian: None,
            preconditioner: None,
            linear_solver: None,
        }
    }
    /// Creates a problem whose Jacobian-vector products come from `jacobian`, which
    /// also supplies the residual. Use it for analytic or automatic-differentiation
    /// products; [`crate::ResidualFn::with_jvp`] wraps them as a closure.
    pub fn with_jacobian(jacobian: &'a dyn JacobianOperator<T>) -> Self {
        Self {
            scaling: None,
            residual: jacobian,
            jacobian: Some(jacobian),
            preconditioner: None,
            linear_solver: None,
        }
    }
    /// Adds a right preconditioner for GMRES. It may change between applications,
    /// as in flexible GMRES.
    pub fn with_preconditioner(mut self, preconditioner: &'a dyn Preconditioner<T>) -> Self {
        self.preconditioner = Some(preconditioner);
        self
    }
    /// Replaces GMRES by `solver` for the Newton linear systems. A problem with both
    /// a linear solver and a preconditioner is rejected with `InvalidInput` when the
    /// workspace is sized or the solve starts.
    pub fn with_linear_solver(mut self, solver: &'a dyn NonlinearLinearSolver<T>) -> Self {
        self.linear_solver = Some(solver);
        self
    }
    /// Adds row and variable divisors (see [`NonlinearScaling`]). Without scaling,
    /// unscaled 2-norms are used.
    pub fn with_scaling(mut self, scaling: &'a dyn NonlinearScaling<T>) -> Self {
        self.scaling = Some(scaling);
        self
    }
    /// Returns the number of unknowns `n`.
    pub fn dimension(&self) -> usize {
        self.residual.input_dimension()
    }
    /// Validate this problem as the `problem` argument.
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        let n = self.dimension();
        let problem = scioxide_core::argument!("problem");
        if self.linear_solver.is_some() && self.preconditioner.is_some() {
            return Err(invalid(
                problem,
                InvalidInputReason::Inconsistent,
                "an injected solver owns its preconditioner",
            ));
        }
        if n == 0 || self.residual.output_dimension() != n {
            return Err(invalid(
                problem,
                InvalidInputReason::Inconsistent,
                "a nonempty square residual map is required",
            ));
        }
        if self.preconditioner.is_some_and(|p| p.dimension() != n) {
            return Err(invalid(
                problem,
                InvalidInputReason::Inconsistent,
                "preconditioner dimension must match the problem",
            ));
        }
        Ok(())
    }
}
