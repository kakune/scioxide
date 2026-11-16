//! Traits for objectives, constraints, least-squares problems, line searches
//! and solvers.

use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::backend::try_filled;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;

use crate::LineSearchResult;
use crate::result::OptimizeResult;

use crate::validation::slice_shape as validate_slice_output;

use crate::validation::input as validate_callback_input;

use crate::validation::matrix_shape as validate_matrix_shape;

use crate::validation::finite_callback as validate_callback_output;

// ---------------------------------------------------------------------------
// Problem traits
// ---------------------------------------------------------------------------

/// Defines a scalar objective `f: ℝⁿ → ℝ` to minimize.
///
/// Implement [`dim`](ObjectiveFunction::dim) and
/// [`eval`](ObjectiveFunction::eval). The derivative methods are optional. An
/// `*_into` method that returns `Ok(false)` says the derivative is not
/// available, and the solver then uses the next source allowed by its
/// [`DerivativeStrategy`] (for example forward differences). The default
/// implementations check shapes and return `Ok(false)`. The methods without
/// `_into` allocate their output and call the `*_into` form.
///
/// An error returned by a callback stops the solve and is returned unchanged.
/// A NaN or infinite value is a `NonFinite` error with origin `Callback`, except
/// at trial points, as described under [Termination](crate#termination).
pub trait ObjectiveFunction<S: RealScalar + NumericScalar<Real = S>> {
    /// Returns whether a finite trial point lies in the domain of `f`.
    ///
    /// Line searches, [`Sqp`](crate::Sqp), [`InteriorPoint`](crate::InteriorPoint),
    /// [`Cobyla`](crate::Cobyla) and [`Cobyqa`](crate::Cobyqa) call this before
    /// [`Self::eval`] and reject a trial for which it returns `false`. The
    /// default checks the length of `x` and returns `true`. Override it for an
    /// objective defined on a subset of `ℝⁿ`, such as one with a logarithmic
    /// barrier.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x.len() != dim()`.
    fn is_in_domain(&self, x: &[S]) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        Ok(true)
    }

    /// Returns the number of variables `n`.
    fn dim(&self) -> usize;

    /// Returns the box bounds of the problem, or `None` if it is unbounded.
    ///
    /// [`crate::BoundedObjective`] attaches a checked box to an objective.
    /// Bound-constrained methods such as [`crate::LbfgsB`] and [`crate::Powell`]
    /// respect it. Methods without bound handling reject a bounded problem
    /// before any callback instead of ignoring the box.
    fn bounds(&self) -> Option<&scioxide_core::Bounds<S>> {
        None
    }

    /// Evaluates `f(x)`.
    ///
    /// # Errors
    ///
    /// Implementations return `DimensionMismatch` if `x.len() != dim()`, and
    /// may return any error of their own; the solver passes it through.
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError>;

    /// Writes the analytic gradient `∇f(x)` to `output`, or returns `Ok(false)`
    /// if it is not available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` does not have
    /// length `dim()`, and `NonFinite` if `x` is not finite.
    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_slice_output(self.dim(), output)?;
        Ok(false)
    }

    /// Returns the analytic gradient, or `None` if it is not available.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::gradient_into`], `AllocationFailure`, and
    /// `NonFinite` if the gradient contains NaN or infinity.
    fn gradient(&self, x: &[S]) -> Result<Option<Vec<S>>, SciOxideError> {
        let mut output = try_filled(
            self.dim(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("objective_gradient_output"),
        )?;
        if !self.gradient_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(&output)?;
        Ok(Some(output))
    }

    /// Writes a gradient computed by automatic differentiation to `output`, or
    /// returns `Ok(false)` if none is available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` does not have
    /// length `dim()`, and `NonFinite` if `x` is not finite.
    fn automatic_gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_slice_output(self.dim(), output)?;
        Ok(false)
    }

    /// Returns a gradient computed by automatic differentiation, or `None`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::automatic_gradient_into`],
    /// `AllocationFailure`, and `NonFinite` if the gradient is not finite.
    fn automatic_gradient(&self, x: &[S]) -> Result<Option<Vec<S>>, SciOxideError> {
        let mut output = try_filled(
            self.dim(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("automatic_objective_gradient_output"),
        )?;
        if !self.automatic_gradient_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(&output)?;
        Ok(Some(output))
    }

    /// Returns whether [`Self::hessian_vector_product_into`] is implemented.
    ///
    /// The answer must not change between calls during a solve. The default is
    /// `false`.
    fn has_hessian_vector_product(&self) -> bool {
        false
    }

    /// Writes the Hessian-vector product `∇²f(x) d` to `output`, where `d` is
    /// `direction`, without forming the Hessian.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` or `NonFinite` for invalid
    /// arguments, and otherwise `InvalidInput` (`UnsupportedConfiguration`).
    fn hessian_vector_product_into(
        &self,
        x: &[S],
        direction: &[S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_callback_input(self.dim(), direction)?;
        validate_slice_output(self.dim(), output)?;
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "objective has no Hessian-vector product",
        ))
    }

    /// Writes the analytic Hessian `∇²f(x)` to the `n × n` matrix `output`, or
    /// returns `Ok(false)` if it is not available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` has the wrong
    /// shape, and `NonFinite` if `x` is not finite.
    fn hessian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_matrix_shape(self.dim(), self.dim(), output)?;
        Ok(false)
    }

    /// Returns the analytic Hessian, or `None` if it is not available.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::hessian_into`], `AllocationFailure`, and
    /// `NonFinite` if the Hessian contains NaN or infinity.
    fn hessian(&self, x: &[S]) -> Result<Option<Matrix<S>>, SciOxideError> {
        let mut output = Matrix::zeros(self.dim(), self.dim())?;
        if !self.hessian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(output.as_slice())?;
        Ok(Some(output))
    }

    /// Writes a Hessian computed by automatic differentiation to `output`, or
    /// returns `Ok(false)` if none is available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` has the wrong
    /// shape, and `NonFinite` if `x` is not finite.
    fn automatic_hessian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_matrix_shape(self.dim(), self.dim(), output)?;
        Ok(false)
    }

    /// Returns a Hessian computed by automatic differentiation, or `None`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::automatic_hessian_into`],
    /// `AllocationFailure`, and `NonFinite` if the Hessian is not finite.
    fn automatic_hessian(&self, x: &[S]) -> Result<Option<Matrix<S>>, SciOxideError> {
        let mut output = Matrix::zeros(self.dim(), self.dim())?;
        if !self.automatic_hessian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(output.as_slice())?;
        Ok(Some(output))
    }
}

/// Defines one scalar constraint function `c: ℝⁿ → ℝ`.
///
/// Add constraint functions to a [`crate::Constraints`] collection, which sets
/// whether each one is an equality `c(x) = 0` or an inequality. As with
/// [`ObjectiveFunction`], a Jacobian method that returns `Ok(false)` says the
/// derivative is not available, and errors from these methods are returned
/// unchanged.
pub trait ConstraintFunction<S: RealScalar + NumericScalar<Real = S>> {
    /// Returns the number of variables `n`.
    fn dim(&self) -> usize;

    /// Evaluates `c(x)`.
    ///
    /// # Errors
    ///
    /// Implementations return `DimensionMismatch` if `x.len() != dim()`, and
    /// may return any error of their own; the solver passes it through.
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError>;

    /// Writes the analytic gradient `∇c(x)` (one Jacobian row) to `output`, or
    /// returns `Ok(false)` if it is not available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` does not have
    /// length `dim()`, and `NonFinite` if `x` is not finite.
    fn jacobian_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_slice_output(self.dim(), output)?;
        Ok(false)
    }

    /// Returns the analytic Jacobian row, or `None` if it is not available.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::jacobian_into`], `AllocationFailure`, and
    /// `NonFinite` if the row contains NaN or infinity.
    fn jacobian(&self, x: &[S]) -> Result<Option<Vec<S>>, SciOxideError> {
        let mut output = try_filled(
            self.dim(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("constraint_jacobian_output"),
        )?;
        if !self.jacobian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(&output)?;
        Ok(Some(output))
    }

    /// Writes a Jacobian row computed by automatic differentiation to `output`,
    /// or returns `Ok(false)` if none is available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` does not have
    /// length `dim()`, and `NonFinite` if `x` is not finite.
    fn automatic_jacobian_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        validate_callback_input(self.dim(), x)?;
        validate_slice_output(self.dim(), output)?;
        Ok(false)
    }

    /// Returns a Jacobian row computed by automatic differentiation, or `None`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::automatic_jacobian_into`],
    /// `AllocationFailure`, and `NonFinite` if the row is not finite.
    fn automatic_jacobian(&self, x: &[S]) -> Result<Option<Vec<S>>, SciOxideError> {
        let mut output = try_filled(
            self.dim(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("automatic_constraint_jacobian_output"),
        )?;
        if !self.automatic_jacobian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(&output)?;
        Ok(Some(output))
    }
}

// ---------------------------------------------------------------------------
// Solver traits
// ---------------------------------------------------------------------------

/// Minimizes a problem of type `F` with a configured solver chosen at run time.
///
/// Every local, constrained and global minimizer that returns an
/// [`OptimizeResult`] implements this trait, so solvers of one problem type
/// can be boxed as `dyn Optimizer<S, F>` and swapped. Method-specific
/// diagnostics are in [`OptimizeResult::method_diagnostics`] and the method
/// name in [`OptimizeResult::algorithm`]. The problem type `F` defaults to
/// `dyn ObjectiveFunction<S>`. Bound-constrained and constrained methods
/// implement the trait for [`crate::BoundedObjective`] and
/// [`crate::ConstrainedProblem`] instead, so
/// `dyn Optimizer<S, ConstrainedProblem<'_, S>>` selects among the constrained
/// methods. For repeated solves, use the solver's `minimize_into`, as described
/// in [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_optimize::{Bfgs, NelderMead, ObjectiveFn, Optimizer};
/// let sphere = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
/// let solvers: [Box<dyn Optimizer<f64>>; 2] =
///     [Box::new(Bfgs::default()), Box::new(NelderMead::default())];
/// for solver in &solvers {
///     let result = solver.minimize(&sphere, &[1.0])?;
///     assert!(result.x()[0].abs() < 1e-4);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// Each solver exposes an inherent `minimize` for owned calls; this trait
/// supports runtime selection through `dyn Optimizer`.
pub trait Optimizer<S: RealScalar + NumericScalar<Real = S>, F: ?Sized = dyn ObjectiveFunction<S>> {
    /// Minimizes `f` starting from `x0` on the calling thread.
    ///
    /// The call allocates a workspace and has no clock, so a configured time
    /// limit fails with `CapabilityUnavailable` before any callback. A run that
    /// stops early with a usable point returns `Ok` with an unsuccessful
    /// [termination](crate#termination).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the start or the problem is incompatible with the
    ///   solver, for example a bounded problem passed to a method without bound
    ///   handling.
    /// - `DimensionMismatch` if `x0.len()` differs from the problem dimension.
    /// - `ConvergenceFailure` or `ExecutionStopped` if the solver stops before
    ///   it has an accepted point.
    /// - Any error returned by a callback.
    fn minimize(&self, f: &F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError>;
}

/// Chooses a step length along a descent direction.
///
/// Given a point `x`, a direction `d` and the directional derivative
/// `∇f(x)ᵀ d < 0`, a line search finds a step `α > 0` such that `x + α d`
/// satisfies its acceptance conditions, for example the Armijo condition
/// ([`crate::BacktrackingLineSearch`]) or the strong Wolfe conditions
/// ([`crate::StrongWolfeLineSearch`]). Implement this trait to supply your own
/// search to a line-search method.
pub trait LineSearch<S: RealScalar + NumericScalar<Real = S>>: core::fmt::Debug {
    /// Searches along `direction` from `x` and writes the chosen trial point.
    ///
    /// `f_x` is `f(x)` and `grad_dot_dir` is `∇f(x)ᵀ d`. `derivative_strategy`
    /// selects how trial gradients are computed. The accepted trial, or the
    /// best trial of an unsuccessful search, is written to `trial_point`, and
    /// its gradient to `trial_gradient` when one was computed.
    /// `candidate_point`, `candidate_gradient` and `finite_difference` are
    /// scratch. `initial_step` is an optional first trial step; the built-in
    /// searches clamp it to their minimum and initial step, and use their
    /// configured initial step for `None`.
    ///
    /// Count each trial through `scope`, so that it counts toward the
    /// evaluation limit of the enclosing solve. A scope made from an
    /// [`scioxide_core::ExecutionContext`] runs the search on its own. The result
    /// says whether the search succeeded. On error, `trial_point` and
    /// `trial_gradient` are unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the configuration is invalid or `direction` is not a
    ///   descent direction.
    /// - `DimensionMismatch` if an input or a returned derivative has the
    ///   wrong length.
    /// - `NonFinite` if `f_x` or a trial gradient is not finite. A non-finite
    ///   trial value only rejects that trial.
    /// - `NumericalInstability` if the trial steps cannot be computed in
    ///   floating point.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, SciOxideError};
    /// use scioxide_differentiate::{DerivativeOperation, DerivativeStrategy};
    /// use scioxide_optimize::{
    ///     BacktrackingLineSearch, DerivativeWorkspace, LineSearch, ObjectiveFunction,
    /// };
    /// struct Parabola;
    /// impl ObjectiveFunction<f64> for Parabola {
    ///     fn dim(&self) -> usize { 1 }
    ///     fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> { Ok(x[0] * x[0]) }
    /// }
    /// # fn main() -> Result<(), SciOxideError> {
    /// let (mut trial, mut gradient) = ([0.0], [0.0]);
    /// let (mut candidate, mut candidate_gradient) = ([0.0], [0.0]);
    /// let mut derivatives = DerivativeWorkspace::new(DerivativeOperation::Gradient, 1, 1)?;
    /// let result = BacktrackingLineSearch::<f64>::default().search_into(
    ///     &Parabola, &[1.0], 1.0, &[-1.0], -2.0, DerivativeStrategy::default(),
    ///     &mut trial, &mut gradient, &mut candidate, &mut candidate_gradient,
    ///     &mut derivatives, ExecutionContext::sequential().into(),
    ///     None,
    /// )?;
    /// assert!(result.is_success());
    /// assert_eq!(trial, [0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn search_into<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x: &[S],
        f_x: S,
        direction: &[S],
        grad_dot_dir: S,
        derivative_strategy: DerivativeStrategy,
        trial_point: &mut [S],
        trial_gradient: &mut [S],
        candidate_point: &mut [S],
        candidate_gradient: &mut [S],
        finite_difference: &mut DerivativeWorkspace<S>,
        scope: crate::LineSearchScope<'_>,
        initial_step: Option<S>,
    ) -> Result<LineSearchResult<S>, SciOxideError>;
}

// ---------------------------------------------------------------------------
// Least-squares problem trait
// ---------------------------------------------------------------------------

/// Defines a nonlinear least-squares problem: minimize `½ ‖r(x)‖²`.
///
/// `r: ℝⁿ → ℝᵐ` is the residual vector, with `n = param_dim()` and
/// `m = residual_dim()`. Implement [`Self::residual_into`]. With optional
/// per-observation scales `sᵢ` and weights `wᵢ`, and the loss `ρ` of
/// [`crate::LeastSquaresLoss`], the cost is
///
/// ```text
/// C(x) = Σᵢ wᵢ ρ(|rᵢ(x)| / sᵢ),   ρ(z) = z²/2 for the quadratic loss,
/// ```
///
/// where `|rᵢ|` is the absolute value of a real residual or the modulus of a
/// complex pair (see [`Self::residual_group_size`]). The Jacobian is optional:
/// a method that returns `Ok(false)` says it is not available, and the solver
/// uses finite differences. Large problems can supply Jacobian-vector products
/// instead of a dense Jacobian.
pub trait LeastSquaresProblem<S: RealScalar + NumericScalar<Real = S>> {
    /// Returns the number of parameters `n`.
    fn param_dim(&self) -> usize;

    /// Returns the number of residual components `m`.
    fn residual_dim(&self) -> usize;

    /// Returns a fixed positive scale for parameter `index`, or `None` for
    /// automatic scaling.
    ///
    /// With `None`, dense solvers scale each parameter by the inverse of the
    /// largest weighted Jacobian column norm seen so far. Problems that supply
    /// only Jacobian products use unit scales.
    fn parameter_scale(&self, _index: usize) -> Option<S> {
        None
    }

    /// Returns the positive scale `sᵢ` of observation `observation`; the
    /// default is 1.
    fn residual_scale(&self, _observation: usize) -> S {
        <S as scioxide_core::AlgebraicScalar>::ONE
    }

    /// Returns the finite nonnegative weight `wᵢ` of observation
    /// `observation`; the default is 1. A zero weight excludes the observation.
    fn residual_weight(&self, _observation: usize) -> S {
        <S as scioxide_core::AlgebraicScalar>::ONE
    }

    /// Returns the number of residual components per observation: 1 for real
    /// residuals (the default), or 2 when consecutive pairs are the real and
    /// imaginary parts of one complex residual.
    fn residual_group_size(&self) -> usize {
        1
    }

    /// Returns the scratch length needed by [`Self::jacobian_product_into`] and
    /// [`Self::jacobian_adjoint_product_into`], or `None` (the default) if the
    /// problem does not supply Jacobian products.
    ///
    /// # Errors
    ///
    /// Implementations may return a size error, such as `SizeOverflow`.
    fn jacobian_operator_workspace_len(&self) -> Result<Option<usize>, SciOxideError> {
        Ok(None)
    }

    /// Writes the Jacobian-vector product `J(x) v` to `output`, where `v` is
    /// `input`, without forming `J`.
    ///
    /// # Errors
    ///
    /// The default returns `InvalidInput` (not implemented). Implementations
    /// return `DimensionMismatch` for wrong lengths and may return their own
    /// errors.
    fn jacobian_product_into(
        &self,
        _x: &[S],
        _input: &[S],
        _output: &mut [S],
        _scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented(
            "least-squares Jacobian product",
        ))
    }

    /// Writes the transposed product `J(x)ᵀ u` to `output`, where `u` is
    /// `input`, without forming `J`.
    ///
    /// # Errors
    ///
    /// The default returns `InvalidInput` (not implemented). Implementations
    /// return `DimensionMismatch` for wrong lengths and may return their own
    /// errors.
    fn jacobian_adjoint_product_into(
        &self,
        _x: &[S],
        _input: &[S],
        _output: &mut [S],
        _scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented(
            "least-squares adjoint product",
        ))
    }

    /// Writes the residual vector `r(x)` to `output`, of length `m`.
    ///
    /// # Errors
    ///
    /// Implementations return `DimensionMismatch` if `x.len() != param_dim()`,
    /// and may return any error of their own; the solver passes it through.
    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError>;

    /// Returns the residual vector `r(x)` in a new vector.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `x.len() != param_dim()`, the errors of
    /// [`Self::residual_into`], `AllocationFailure`, and `NonFinite` if a
    /// residual is NaN or infinite.
    fn residual(&self, x: &[S]) -> Result<Vec<S>, SciOxideError> {
        validate_callback_input(self.param_dim(), x)?;
        let mut output = try_filled(
            self.residual_dim(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("least_squares_residual_output"),
        )?;
        self.residual_into(x, &mut output)?;
        validate_callback_output(&output)?;
        Ok(output)
    }

    /// Writes the `m × n` Jacobian `J(x)`, with `Jᵢⱼ = ∂rᵢ/∂xⱼ`, to `output`,
    /// or returns `Ok(false)` if it is not available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` has the wrong
    /// shape, and `NonFinite` if `x` is not finite.
    fn jacobian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        validate_callback_input(self.param_dim(), x)?;
        validate_matrix_shape(self.residual_dim(), self.param_dim(), output)?;
        Ok(false)
    }

    /// Returns the analytic Jacobian `J(x)`, or `None` if it is not available.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::jacobian_into`], `AllocationFailure`, and
    /// `NonFinite` if the Jacobian contains NaN or infinity.
    fn jacobian(&self, x: &[S]) -> Result<Option<Matrix<S>>, SciOxideError> {
        let mut output = Matrix::zeros(self.residual_dim(), self.param_dim())?;
        if !self.jacobian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(output.as_slice())?;
        Ok(Some(output))
    }

    /// Writes a Jacobian computed by automatic differentiation to `output`, or
    /// returns `Ok(false)` if none is available.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `x` or `output` has the wrong
    /// shape, and `NonFinite` if `x` is not finite.
    fn automatic_jacobian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        validate_callback_input(self.param_dim(), x)?;
        validate_matrix_shape(self.residual_dim(), self.param_dim(), output)?;
        Ok(false)
    }

    /// Returns a Jacobian computed by automatic differentiation, or `None`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::automatic_jacobian_into`],
    /// `AllocationFailure`, and `NonFinite` if the Jacobian is not finite.
    fn automatic_jacobian(&self, x: &[S]) -> Result<Option<Matrix<S>>, SciOxideError> {
        let mut output = Matrix::zeros(self.residual_dim(), self.param_dim())?;
        if !self.automatic_jacobian_into(x, &mut output)? {
            return Ok(None);
        }
        validate_callback_output(output.as_slice())?;
        Ok(Some(output))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    // ---- ObjectiveFunction ----

    struct SimpleQuadratic;

    impl ObjectiveFunction<f64> for SimpleQuadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(true)
        }
    }

    struct NoGradFn;

    impl ObjectiveFunction<f64> for NoGradFn {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0])
        }
    }

    #[test]
    fn test_objective_function_eval() {
        let f = SimpleQuadratic;
        let val = f.eval(&[1.0, 2.0]);
        assert!(val.is_ok());
        assert!((val.unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_objective_function_gradient() {
        let f = SimpleQuadratic;
        let g = f.gradient(&[3.0, 4.0]);
        assert!(g.is_ok());
        let g = g.unwrap().unwrap();
        assert!((g[0] - 6.0).abs() < 1e-12);
        assert!((g[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_objective_function_default_gradient() {
        let f = NoGradFn;
        let g = f.gradient(&[1.0]);
        assert!(g.is_ok());
        assert!(g.unwrap().is_none());
    }

    #[test]
    fn test_objective_function_default_hessian() {
        let f = NoGradFn;
        let h = f.hessian(&[1.0]);
        assert!(h.is_ok());
        assert!(h.unwrap().is_none());
    }

    #[test]
    fn test_objective_function_dyn_dispatch() {
        let f: &dyn ObjectiveFunction<f64> = &SimpleQuadratic;
        assert_eq!(f.dim(), 2);
        let val = f.eval(&[1.0, 0.0]);
        assert!(val.is_ok());
        assert!((val.unwrap() - 1.0).abs() < 1e-12);
    }

    // ---- ConstraintFunction ----

    struct LinearConstraint;

    impl ConstraintFunction<f64> for LinearConstraint {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] + x[1] - 1.0)
        }
        fn jacobian_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[1.0, 1.0]);
            Ok(true)
        }
    }

    struct NoJacConstraint;

    impl ConstraintFunction<f64> for NoJacConstraint {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] - 1.0)
        }
    }

    #[test]
    fn test_constraint_function_eval() {
        let c = LinearConstraint;
        let val = c.eval(&[0.5, 0.5]);
        assert!(val.is_ok());
        assert!((val.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_constraint_function_jacobian() {
        let c = LinearConstraint;
        let j = c.jacobian(&[1.0, 2.0]);
        assert!(j.is_ok());
        let j = j.unwrap().unwrap();
        assert!((j[0] - 1.0).abs() < 1e-12);
        assert!((j[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_constraint_function_default_jacobian() {
        let c = NoJacConstraint;
        let j = c.jacobian(&[1.0]);
        assert!(j.is_ok());
        assert!(j.unwrap().is_none());
    }

    #[test]
    fn test_constraint_function_dyn_dispatch() {
        let c: &dyn ConstraintFunction<f64> = &LinearConstraint;
        let val = c.eval(&[1.0, 0.0]);
        assert!(val.is_ok());
        assert!((val.unwrap()).abs() < 1e-12);
    }

    // ---- LeastSquaresProblem ----

    struct SimpleLSP;

    impl LeastSquaresProblem<f64> for SimpleLSP {
        fn param_dim(&self) -> usize {
            2
        }
        fn residual_dim(&self) -> usize {
            3
        }
        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]);
            Ok(())
        }
    }

    #[test]
    fn test_least_squares_residual() {
        let p = SimpleLSP;
        assert_eq!(p.param_dim(), 2);
        assert_eq!(p.residual_dim(), 3);
        let r = p.residual(&[1.0, 2.0]);
        assert!(r.is_ok());
        let r = r.unwrap();
        assert!((r[0]).abs() < 1e-12);
        assert!((r[1]).abs() < 1e-12);
        assert!((r[2]).abs() < 1e-12);
    }

    #[test]
    fn test_least_squares_default_jacobian() {
        let p = SimpleLSP;
        let j = p.jacobian(&[1.0, 2.0]);
        assert!(j.is_ok());
        assert!(j.unwrap().is_none());
    }
}
