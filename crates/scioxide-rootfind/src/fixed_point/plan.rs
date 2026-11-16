//! Fixed-point solver and its one-shot form.
use super::*;
use crate::attempt::{Conclusion, Entry};

/// Solves `x = G(x)` or `F(x) = 0` for `x ∈ Tⁿ` by fixed-point iteration with
/// optional acceleration.
///
/// The [`FixedPointInput`] says whether the residual operator returns `G(x)` or
/// `F(x)`; the [`FixedPointMethod`] selects damped Picard iteration, Steffensen's
/// method, Anderson acceleration or a Broyden-type update. `T` is a real or
/// complex scalar. Each iteration costs one residual evaluation (two for
/// Steffensen) and `O(n m)` work for `m` history pairs, plus an `O(n m²)` QR
/// factorization for Anderson. The solver needs no Jacobian. For a system whose
/// Jacobian or Jacobian-vector products are available, Newton-type solvers such
/// as [`NewtonKrylovSolver`](crate::NewtonKrylovSolver) usually converge in fewer
/// evaluations.
///
/// The solver object holds validated options; the history lives in a
/// [`FixedPointWorkspace`]. Success and stopping rules are described on
/// [`FixedPointOptions`] and in [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{
///     FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointSolver, ResidualFn,
/// };
///
/// // G(x) = x/2 + 1 has the fixed point x = 2.
/// let map = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = 0.5 * x[0] + 1.0;
///     Ok(())
/// });
/// let options = FixedPointOptions::default();
/// let solver = FixedPointSolver::new(FixedPointMethod::Anderson, FixedPointInput::Map, options)?;
/// let result = solver.solve(&map, &[0.0])?;
/// assert!(result.is_success());
/// assert!((result.point()[0] - 2.0).abs() < 1e-7);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct FixedPointSolver<T: NumericScalar> {
    pub(crate) method: FixedPointMethod,
    pub(crate) input: FixedPointInput,
    pub(crate) options: FixedPointOptions<T::Real>,
    pub(crate) scalar: core::marker::PhantomData<T>,
}

impl<T: NumericScalar> FixedPointSolver<T> {
    /// Creates a solver after validating `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the damping, maximum mixing coefficient, residual scale
    ///   or a component divisor is not positive and finite, if the maximum mixing
    ///   coefficient is below the damping, if the history capacity is zero, or if
    ///   the regularization is negative or not finite.
    /// - `InvalidInput` if `method` is [`FixedPointMethod::ExcitingMixing`] and `T`
    ///   is complex.
    pub fn new(
        method: FixedPointMethod,
        input: FixedPointInput,
        options: FixedPointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if method == FixedPointMethod::ExcitingMixing
            && T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component()
                != T::Real::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("method"),
                InvalidInputReason::Inconsistent,
                "exciting mixing supports real scalars only",
            ));
        }
        Ok(Self {
            method,
            input,
            options,
            scalar: core::marker::PhantomData,
        })
    }
    /// Returns the method.
    pub const fn method(&self) -> FixedPointMethod {
        self.method
    }
    /// Returns how the residual operator output is read.
    pub const fn input(&self) -> FixedPointInput {
        self.input
    }
    /// Returns the validated options.
    pub const fn options(&self) -> &FixedPointOptions<T::Real> {
        &self.options
    }
    /// Returns the workspace size needed for `problem`, including the scratch
    /// length that the residual operator requests.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `problem` is not square.
    /// - `InvalidInput` if the number of component divisors differs from the
    ///   problem dimension.
    /// - `SizeOverflow` if the size does not fit in `usize`, and `ResourceLimit` if
    ///   it exceeds [`FixedPointOptions::with_workspace_byte_limit`].
    /// - Errors from the residual operator's scratch-size query.
    pub fn workspace_requirements<F: ResidualOperator<T> + ?Sized>(
        &self,
        problem: &F,
    ) -> Result<FixedPointWorkspaceRequirements<T>, SciOxideError> {
        let n = problem.input_dimension();
        if problem.output_dimension() != n {
            return Err(SciOxideError::dimension_mismatch(
                [n],
                [problem.output_dimension()],
            ));
        }
        if !self.options.component_scales.is_empty() && self.options.component_scales.len() != n {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                InvalidInputReason::Inconsistent,
                "component scales must match dimension",
            ));
        }
        FixedPointWorkspaceRequirements::<T>::new(
            self.method,
            n,
            problem.residual_workspace_len()?,
            &self.options,
        )
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure` if
    /// memory cannot be reserved.
    pub fn workspace<F: ResidualOperator<T> + ?Sized>(
        &self,
        problem: &F,
    ) -> Result<FixedPointWorkspace<T>, SciOxideError> {
        let r = self.workspace_requirements(problem)?;
        FixedPointWorkspace::allocate(&r)
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
    pub fn solve<F: ResidualOperator<T> + ?Sized>(
        &self,
        problem: &F,
        initial: &[T],
    ) -> Result<FixedPointResult<T>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut result = FixedPointResult::new(initial.len())?;
        self.solve_into(
            problem,
            initial,
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(result)
    }
    /// Solves from `initial` into `output`, using `workspace` for the history.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or the
    /// [`SolveContext`](scioxide_core::SolveContext) of an enclosing solve. When a
    /// budget, cancellation or deadline stops the solve after the initial point has
    /// been evaluated, `output` holds the last accepted point with an unsuccessful
    /// [`RootTermination`]. On error, `output` is unchanged
    /// (see [Output on error](scioxide_core#output-on-error)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `problem` is not square, or if `initial` or
    ///   `output` does not have the problem dimension.
    /// - `InvalidInput` if `workspace` was not prepared for this solver and
    ///   problem, or if the number of component divisors is wrong.
    /// - `NonFinite` if `initial` or a residual value is NaN or infinite.
    /// - `InvalidInput` if the regularized Anderson least-squares matrix is still
    ///   rank deficient.
    /// - `ExecutionStopped` or `ConvergenceFailure` if a stop occurs before the
    ///   initial point is evaluated.
    /// - Errors returned by the residual operator, unchanged.
    pub fn solve_into<'scope, 'capability, F: ResidualOperator<T> + ?Sized>(
        &self,
        problem: &F,
        initial: &[T],
        output: &mut FixedPointResult<T>,
        workspace: &mut FixedPointWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::attempt(
            workspace,
            output,
            scope.into(),
            scioxide_core::AttemptKind::Call,
            |workspace, output| {
                let r = self.workspace_requirements(problem)?;
                if initial.len() != r.dimension || output.point.len() != r.dimension {
                    return Err(SciOxideError::dimension_mismatch(
                        [r.dimension],
                        [initial.len()],
                    ));
                }
                crate::residual::finite(initial, NonFiniteOrigin::Input)?;
                if workspace.check(&r)? != scioxide_core::Capacity::Sufficient {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "reserve matching structural requirements",
                    ));
                }
                workspace.accepted = false;
                Ok(Entry::call(
                    scioxide_core::SolveIdentity::new(
                        scioxide_core::ProblemToken::new(0),
                        scioxide_core::algorithm_id!("rootfind", "FixedPoint", 1),
                        r.dimension as u128,
                        0,
                    ),
                    crate::context::cap(
                        self.options.budgets,
                        scioxide_core::BudgetKind::Iteration,
                        self.options.iterations,
                    ),
                ))
            },
            |workspace, output, context| {
                self.kernel(problem, initial, &self.options, output, workspace, context)
            },
            |workspace, output, _| Conclusion {
                succeeded: output.is_success(),
                progress: workspace.accepted.then_some(workspace.diagnostics),
            },
        )
    }
}

/// Finds a fixed point `x = G(x)` of `g` from `initial` by Anderson
/// acceleration and returns it.
///
/// `g` writes `G(x) ∈ Tⁿ` of `x ∈ Tⁿ` to its output slice, where
/// `n = initial.len()` and `T` is a real or complex scalar. The iteration is
/// that of [`FixedPointSolver`] with [`FixedPointMethod::Anderson`],
/// [`FixedPointInput::Map`] and the default [`FixedPointOptions`]: type-II
/// Anderson acceleration of the residual `f(x) = G(x) − x` with at most
/// `m = min(5, n)` history pairs (the oldest pair is dropped first), damping 1
/// and no regularization. Each iteration costs one evaluation of `g` and
/// `O(n m²)` operations for the column-pivoted QR factorization of the
/// least-squares problem over the history.
///
/// The solve succeeds when `‖G(x) − x‖₂ ≤ √ε·‖G(x₀) − x₀‖₂` within 200
/// iterations, where `x₀` is `initial` and `ε` the machine epsilon of the real
/// type of `T`. The returned point equals the point of the configured solve bit
/// for bit, and an empty `initial` gives an empty vector. The call allocates one
/// workspace and one result.
///
/// Use [`FixedPointSolver`] with [`FixedPointOptions`] for another method,
/// damping, history, tolerances or budgets, for a residual `F(x) = 0`
/// ([`FixedPointInput::Residual`]), or to read the residual and work counts.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if a component of `initial` is NaN or
///   infinite.
/// - `NonFinite` (origin `Callback`) if `g` writes NaN or infinity; errors
///   returned by `g` are returned unchanged.
/// - `AllocationFailure` if the workspace or result cannot be allocated.
/// - `ConvergenceFailure` (`IterationLimit` after 200 iterations, otherwise
///   `Stagnation`) if the residual test is not met (see the
///   [crate errors](crate#errors)).
///
/// # Examples
///
/// ```
/// // G(x) = x/2 + 1 has the fixed point x = 2.
/// let x = scioxide_rootfind::fixed_point(
///     |x: &[f64], out: &mut [f64]| {
///         out[0] = 0.5 * x[0] + 1.0;
///         Ok(())
///     },
///     &[0.0],
/// )?;
/// assert!((x[0] - 2.0).abs() < 1e-7);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations",
/// *SIAM J. Numer. Anal.* 49(4), 2011, 1715–1735.
pub fn fixed_point<T, G>(g: G, initial: &[T]) -> Result<Vec<T>, SciOxideError>
where
    T: NumericScalar,
    G: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    let solver = FixedPointSolver::<T>::new(
        FixedPointMethod::Anderson,
        FixedPointInput::Map,
        FixedPointOptions::default(),
    )?;
    let result = solver.solve(&crate::ResidualFn::new(initial.len(), g), initial)?;
    crate::termination::require_success(result.termination(), result.accounting())?;
    Ok(result.point.into_vec())
}
