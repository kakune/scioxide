//! Dense nonlinear-system solvers: Newton, Broyden and Powell's hybrid method.
//!
//! The three solvers share input checks, attempt bookkeeping and the
//! [`SystemRootSolver`] implementation. They differ in their kernel and in the
//! workspace layout selected by [`DenseMethod`].
use super::options::*;
use super::workspace::{DenseMethod, SystemWorkspace, SystemWorkspaceRequirements};
use crate::attempt::{Conclusion, Entry};
use crate::initialization::validate_initial;
use crate::residual::square_dimension;
use crate::{
    BacktrackingLineSearch, BroydenUpdate, DenseJacobian, ResidualFn, RootLineSearch,
    SystemRootResult, SystemRootSolver,
};
use alloc::vec::Vec;
use scioxide_core::{
    AlgorithmId, AttemptKind, BudgetKind, ExecutionContext, NumericScalar, RealScalar,
    SciOxideError, SolveContext, SolveScope,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{GmresSolver, IterativeLinearSolver};

/// Checks the inputs and runs one dense-system solve.
///
/// Rejects a function, starting point, output or workspace of the wrong size
/// and a non-finite starting point before the solve starts, clears the accepted
/// state, and records the accepted residual as progress.
#[expect(
    clippy::too_many_arguments,
    reason = "One dense attempt borrows the solver, options, problem, guess, output, storage and scope."
)]
fn solve_dense<'scope, 'capability, S, F>(
    method: DenseMethod,
    algorithm: AlgorithmId,
    settings: &Settings<S>,
    view: &SystemRootOptions<'_, S>,
    function: &F,
    initial: &[S],
    output: &mut SystemRootResult<S>,
    workspace: &mut SystemWorkspace<S>,
    scope: SolveScope<'scope, 'capability>,
    kernel: impl FnOnce(
        &mut SystemWorkspace<S>,
        &mut SystemRootResult<S>,
        &mut SolveContext<'_, 'capability>,
    ) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    crate::attempt::attempt(
        workspace,
        output,
        scope,
        AttemptKind::Call,
        |workspace, output| {
            let dimension = square_dimension(function, scioxide_core::argument!("function"))?;
            validate_initial(dimension, initial)?;
            if output.point().len() != dimension || output.residual().len() != dimension {
                return Err(SciOxideError::dimension_mismatch(
                    [dimension],
                    [output.point().len()],
                ));
            }
            let requirements =
                SystemWorkspaceRequirements::for_solve(function, method, settings, view)?;
            if workspace.check(&requirements)? != scioxide_core::Capacity::Sufficient {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "reserve matching structural requirements before solving",
                ));
            }
            workspace.state.accepted = None;
            Ok(Entry::call(
                scioxide_core::SolveIdentity::new(
                    scioxide_core::ProblemToken::new(0),
                    algorithm,
                    dimension as u128,
                    0,
                ),
                crate::context::cap(settings.budgets, BudgetKind::Iteration, settings.iterations),
            ))
        },
        kernel,
        |workspace, output, _| Conclusion {
            succeeded: output.is_success(),
            progress: workspace.state.accepted,
        },
    )
}

/// Declares the accessors and the [`SystemRootSolver`] implementation of a dense solver.
///
/// `method` selects the storage family and `kernel` builds the numerical kernel
/// from the solver bound to `$this`.
macro_rules! dense_plan {
    (
        $plan:ident<$($param:ident),+>,
        where [$($bound:tt)*],
        options: $options:ty,
        name: $name:literal,
        method: |$m:ident| $method:expr,
        kernel: |$k:ident| $kernel:expr $(,)?
    ) => {
        impl<$($param),+> $plan<$($param),+>
        where
            $($bound)*
        {
            /// Returns the method name.
            pub const fn name(&self) -> &'static str {
                $name
            }
            /// Returns the method identifier recorded in results.
            pub const fn algorithm_id(&self) -> AlgorithmId {
                scioxide_core::algorithm_id!("rootfind", $name, 2)
            }
            /// Returns the options.
            pub const fn options(&self) -> &$options {
                &self.options
            }
            /// Returns how the Jacobian is obtained.
            pub const fn derivative_strategy(&self) -> DerivativeStrategy {
                self.options.settings.strategy
            }
            const fn method(&self) -> DenseMethod {
                let $m = self;
                $method
            }
        }
        impl<$($param),+, F> SystemRootSolver<S, F> for $plan<$($param),+>
        where
            $($bound)*
            F: DenseJacobian<S> + ?Sized,
        {
            type Workspace = SystemWorkspace<S>;
            fn name(&self) -> &'static str {
                $name
            }
            fn algorithm_id(&self) -> AlgorithmId {
                scioxide_core::algorithm_id!("rootfind", $name, 2)
            }
            fn workspace_requirements(
                &self,
                function: &F,
            ) -> Result<SystemWorkspaceRequirements<S>, SciOxideError> {
                SystemWorkspaceRequirements::new(
                    function,
                    self.method(),
                    &self.options.settings,
                    &self.options.view(),
                )
            }
            fn workspace(&self, function: &F) -> Result<Self::Workspace, SciOxideError> {
                let mut workspace = SystemWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(function)?)?;
                Ok(workspace)
            }
            fn solve(
                &self,
                function: &F,
                initial: &[S],
            ) -> Result<SystemRootResult<S>, SciOxideError> {
                let dimension = square_dimension(function, scioxide_core::argument!("function"))?;
                validate_initial(dimension, initial)?;
                let mut workspace = self.workspace(function)?;
                let mut output = SystemRootResult::empty();
                output.try_reserve(dimension)?;
                self.solve_into(
                    function,
                    initial,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            fn solve_into<'scope, 'capability>(
                &self,
                function: &F,
                initial: &[S],
                output: &mut SystemRootResult<S>,
                workspace: &mut Self::Workspace,
                scope: impl Into<SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError> {
                let view = self.options.view();
                let $k = self;
                let kernel = $kernel;
                solve_dense(
                    self.method(),
                    <Self as SystemRootSolver<S, F>>::algorithm_id(self),
                    &self.options.settings,
                    &view,
                    function,
                    initial,
                    output,
                    workspace,
                    scope.into(),
                    |workspace, output, context| {
                        kernel.solve_with_context(
                            function, initial, &view, output, workspace, context,
                        )
                    },
                )
            }
        }
    };
}

/// Solves a square nonlinear system `F(x) = 0` by Newton's method.
///
/// Each iteration solves `J(x) Δx = −F(x)` for the Newton step and sets
/// `x ← x + α Δx`. Without a line search `α = 1`. With a line search
/// ([`NewtonSystemOptions::with_line_search`]) `α` comes from backtracking on
/// the merit function `½‖F(x)‖²`, using the exact slope `Fᵀ J Δx`.
///
/// By default the step comes from an LU factorization with partial pivoting,
/// which costs O(n³) per iteration plus one Jacobian evaluation (n or 2n
/// residual evaluations for a forward or central difference Jacobian).
/// [`NewtonSystemOptions::with_linear_solver`] replaces it with an iterative
/// solver, which always solves the Newton system.
///
/// Without a line search a singular Jacobian is an error. With a line search
/// and the LU solve, a numerically rank-deficient Jacobian gives the perturbed
/// step `(JᵀJ + μI) Δx = −JᵀF` with `μ = √(nε) ‖JᵀJ‖₁` (Dennis and Schnabel,
/// Algorithm A6.5.1), a descent direction for the merit function. If the line
/// search finds no acceptable Newton step, the iteration retries once with the
/// perturbed step before it stops with
/// [`RootTermination::GlobalizationFailure`](crate::RootTermination::GlobalizationFailure).
/// A step that does not strictly decrease the merit function counts as a
/// failed search.
///
/// Newton's method converges quadratically near a root with a nonsingular
/// Jacobian. Termination statuses and stopping tests are described under
/// [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{NewtonSystem, NewtonSystemOptions, ResidualFn, SystemRootSolver};
///
/// // F(x) = x − 2 has the root x = 2.
/// let function = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] - 2.0;
///     Ok(())
/// })
/// .with_jacobian(|_: &[f64], out: &mut [f64]| {
///     out[0] = 1.0;
///     Ok(())
/// });
/// let solver = NewtonSystem::new(NewtonSystemOptions::default())?;
/// let result = solver.solve(&function, &[0.0])?;
/// assert!(result.is_success() && (result.point()[0] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. E. Dennis Jr. and R. B. Schnabel, *Numerical Methods for Unconstrained
/// Optimization and Nonlinear Equations*, SIAM, 1996, chapters 5 and 6.
#[derive(Debug, Clone)]
pub struct NewtonSystem<
    S: RealScalar + NumericScalar<Real = S>,
    L = BacktrackingLineSearch<S>,
    I = GmresSolver<S>,
> {
    options: NewtonSystemOptions<S, L, I>,
}
impl<S, L, I> NewtonSystem<S, L, I>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: RootLineSearch<S>,
    I: IterativeLinearSolver<S>,
{
    /// Creates a Newton solver with the given options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the residual scale is not positive and finite.
    pub fn new(options: NewtonSystemOptions<S, L, I>) -> Result<Self, SciOxideError> {
        options.settings.validate()?;
        Ok(Self { options })
    }
}
dense_plan!(
    NewtonSystem<S, L, I>,
    where [S: RealScalar + NumericScalar<Real = S>, L: RootLineSearch<S>, I: IterativeLinearSolver<S>,],
    options: NewtonSystemOptions<S, L, I>,
    name: "NewtonSystem",
    method: |_plan| DenseMethod::Newton,
    kernel: |plan| super::newton_system::NewtonSystemKernel::new(plan.options.settings.strategy),
);

/// Solves a square nonlinear system `F(x) = 0` by Broyden's quasi-Newton method.
///
/// The method keeps a model `B` of the Jacobian. It evaluates the Jacobian at
/// the initial point and then corrects `B` after each step `s` by a rank-one
/// secant update, with `y = F(x + s) − F(x)` (Dennis and Schnabel, chapter 8):
///
/// ```text
/// First (good) update:  B ← B + (y − B s) sᵀ / (sᵀ s)
/// Second (bad) update:  B⁻¹ ← B⁻¹ + (s − B⁻¹ y) yᵀ / (yᵀ y),
///                       applied to B as B ← B + (y − B s) (Bᵀ y)ᵀ / (yᵀ B s)
/// ```
///
/// [`BroydenUpdate`] selects the formula. The first update keeps `B` and an
/// inverse that is updated alongside it. Each direction from the inverse is
/// checked against `B`, and the inverse is recomputed from an LU factorization
/// when the check fails. The second update keeps `B = QR` and updates the
/// factors by plane rotations. The first update skips a zero step, and the
/// second skips an update whose denominator is at most `√ε` times the norms of
/// its factors. A typical iteration costs O(n²) operations and one residual
/// evaluation, against O(n³) and a Jacobian evaluation for [`NewtonSystem`].
///
/// Each step solves `B d = −F`. Without a line search a rank-deficient model is
/// an error. With a line search ([`BroydenOptions::with_line_search`]), a
/// numerically rank-deficient model gives the perturbed step
/// `(BᵀB + μI) d = −BᵀF` (Dennis and Schnabel, Algorithm A6.5.1), and the search
/// uses the model slope `Fᵀ B d`. If the search fails with an updated model, the
/// Jacobian is evaluated again at the accepted point and the step is retried. A
/// failed search with a freshly evaluated Jacobian stops with
/// [`RootTermination::GlobalizationFailure`](crate::RootTermination::GlobalizationFailure).
///
/// Broyden's method converges superlinearly near a root with a nonsingular
/// Jacobian. See [Termination](crate#termination) for the stopping tests.
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{Broyden, BroydenOptions, BroydenUpdate, ResidualFn, SystemRootSolver};
///
/// // F(x) = x − 2 has the root x = 2.
/// let function = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] - 2.0;
///     Ok(())
/// })
/// .with_jacobian(|_: &[f64], out: &mut [f64]| {
///     out[0] = 1.0;
///     Ok(())
/// });
/// let solver = Broyden::new(BroydenUpdate::First, BroydenOptions::default())?;
/// assert!(solver.solve(&function, &[0.0])?.is_success());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. G. Broyden, "A class of methods for solving nonlinear simultaneous
///   equations", *Math. Comp.* 19 (1965), 577–593.
/// - J. E. Dennis Jr. and R. B. Schnabel, *Numerical Methods for Unconstrained
///   Optimization and Nonlinear Equations*, SIAM, 1996, chapter 8.
#[derive(Debug, Clone)]
pub struct Broyden<S: RealScalar + NumericScalar<Real = S>, L = BacktrackingLineSearch<S>> {
    options: BroydenOptions<S, L>,
    update: BroydenUpdate,
}
impl<S, L> Broyden<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: RootLineSearch<S>,
{
    /// Creates a Broyden solver with the given update formula and options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the residual scale is not positive and finite.
    pub fn new(
        update: BroydenUpdate,
        options: BroydenOptions<S, L>,
    ) -> Result<Self, SciOxideError> {
        options.settings.validate()?;
        Ok(Self { options, update })
    }
}
dense_plan!(
    Broyden<S, L>,
    where [S: RealScalar + NumericScalar<Real = S>, L: RootLineSearch<S>,],
    options: BroydenOptions<S, L>,
    name: "Broyden",
    method: |plan| match plan.update {
        BroydenUpdate::First => DenseMethod::BroydenFirst,
        BroydenUpdate::Second => DenseMethod::BroydenSecond,
    },
    kernel: |plan| super::broyden::BroydenKernel::with_initial_jacobian(
        plan.update,
        plan.options.settings.strategy,
    ),
);

/// Solves a real square nonlinear system `F(x) = 0` by Powell's hybrid dogleg
/// trust-region method.
///
/// The variables are scaled by a diagonal matrix `D`: each entry is the largest
/// Euclidean norm of the corresponding Jacobian column seen so far, or one for a
/// zero column. Each trial step `p` minimizes the linear model `‖F + J p‖₂` on the
/// dogleg path between the scaled steepest-descent point and the Gauss–Newton
/// point, subject to `‖D p‖₂ ≤ Δ`. The Gauss–Newton point comes from an
/// unpivoted QR factorization of `J`. A zero diagonal entry of `R` is replaced by
/// `ε` times the largest magnitude in its column.
///
/// The initial radius is `Δ = 100 ‖D x₀‖`, or `100 ‖F(x₀)‖` when `D x₀ = 0`, so
/// that it does not depend on the units of `F`. Until the first successful step
/// the radius is reduced to `‖D p‖`. With `ρ` the ratio of actual to predicted
/// reduction of `‖F‖²`:
///
/// - `ρ < 0.1` halves the radius;
/// - `ρ ≥ 0.5`, or a repeated success, grows it to at least `2 ‖D p‖`;
/// - `|ρ − 1| ≤ 0.1` sets it to `2 ‖D p‖`;
/// - steps with `ρ ≥ 10⁻⁴` are accepted.
///
/// After every trial the model is corrected by the scaled Broyden update
/// `J ← J + (ΔF − J p) (D² p)ᵀ / ‖D p‖²`, applied to the QR factors by plane
/// rotations. A fresh Jacobian is evaluated only after two consecutive trials
/// with `ρ < 0.1`. A typical iteration therefore costs O(n²) operations and one
/// residual evaluation.
///
/// The solve succeeds when the residual tolerance is met. It stops with
/// [`RootTermination::Stagnation`](crate::RootTermination::Stagnation) when the
/// radius meets the step tolerance relative to `‖D x‖`, when no representable
/// improvement remains (`0.1 · max(0.1 Δ, ‖D p‖) ≤ ε ‖D x‖`), after five
/// Jacobian evaluations without a 10% reduction of `‖F‖²`, or after ten trials
/// without a 0.1% reduction of `‖F‖²`.
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{ResidualFn, SystemRootSolver, TrustRegionSystem, TrustRegionSystemOptions};
///
/// // F(x) = x − 2 has the root x = 2.
/// let function = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] - 2.0;
///     Ok(())
/// })
/// .with_jacobian(|_: &[f64], out: &mut [f64]| {
///     out[0] = 1.0;
///     Ok(())
/// });
/// let solver = TrustRegionSystem::new(TrustRegionSystemOptions::default())?;
/// assert!(solver.solve(&function, &[0.0])?.is_success());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - M. J. D. Powell, "A hybrid method for nonlinear equations", in
///   P. Rabinowitz (ed.), *Numerical Methods for Nonlinear Algebraic Equations*,
///   Gordon and Breach, 1970, 87–114.
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
///   2006, §11.2.
#[derive(Debug, Clone)]
pub struct TrustRegionSystem<S: RealScalar + NumericScalar<Real = S>> {
    options: TrustRegionSystemOptions<S>,
}
impl<S> TrustRegionSystem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates a trust-region solver with the given options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the residual scale is not positive and finite.
    pub fn new(options: TrustRegionSystemOptions<S>) -> Result<Self, SciOxideError> {
        options.settings.validate()?;
        Ok(Self { options })
    }
}
dense_plan!(
    TrustRegionSystem<S>,
    where [S: RealScalar + NumericScalar<Real = S>,],
    options: TrustRegionSystemOptions<S>,
    name: "TrustRegionSystem",
    method: |_plan| DenseMethod::TrustRegion,
    kernel: |plan| super::trust_region::TrustRegionSystemKernel::new(plan.options.settings.strategy),
);

/// Solves a real square nonlinear system `F(x) = 0` from `initial` by Powell's
/// hybrid dogleg trust-region method and returns the root.
///
/// `f` writes the residual `F(x) ∈ ℝⁿ` of `x ∈ ℝⁿ` to its output slice, where
/// `n = initial.len()`. The iteration is that of [`TrustRegionSystem`] with the
/// default [`TrustRegionSystemOptions`]. The Jacobian comes from forward
/// differences (`n` residual evaluations) at the start and after two
/// consecutive trial steps with poor model agreement; between them a Broyden
/// update corrects it, so a typical iteration costs one residual evaluation and
/// `O(n²)` operations. Every accepted step reduces `‖F‖₂`, so the iteration
/// converges from many starting points where Newton's method without a line
/// search diverges.
///
/// The solve succeeds when `‖F(x)‖₂ ≤ √ε·‖F(x₀)‖₂` within 100 iterations, where
/// `x₀` is `initial` and `ε` the machine epsilon of `S`; a small step alone is
/// not success. The returned point equals the point of the configured solve bit
/// for bit. The call allocates one workspace and one result.
///
/// Use [`TrustRegionSystem`], [`NewtonSystem`] or [`Broyden`] with their options
/// for other tolerances or budgets, for an analytic Jacobian
/// ([`ResidualFn::with_jacobian`]), or to read the residual and work counts.
///
/// # Errors
///
/// - `InvalidInput` (`initial`, `OutOfRange`) if `initial` is empty.
/// - `NonFinite` (origin `Input`) if a component of `initial` is NaN or
///   infinite.
/// - `NonFinite` (origin `Callback`) if `f` writes NaN or infinity; errors
///   returned by `f` are returned unchanged.
/// - `NumericalInstability` if the trust-region model breaks down.
/// - `AllocationFailure` if the workspace or result cannot be allocated.
/// - `ConvergenceFailure` (`IterationLimit` after 100 iterations, otherwise
///   `Stagnation`) if the residual test is not met (see the
///   [crate errors](crate#errors)).
///
/// # Examples
///
/// ```
/// // x² + y − 1 = 0 and x + y² − 1 = 0 meet at x = y = (√5 − 1)/2.
/// let x = scioxide_rootfind::solve_system(
///     |x: &[f64], out: &mut [f64]| {
///         out[0] = x[0] * x[0] + x[1] - 1.0;
///         out[1] = x[0] + x[1] * x[1] - 1.0;
///         Ok(())
///     },
///     &[0.5, 0.5],
/// )?;
/// let root = (5.0_f64.sqrt() - 1.0) / 2.0;
/// assert!(x.iter().all(|x| (x - root).abs() < 1e-7));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. J. D. Powell, "A hybrid method for nonlinear equations", in
/// P. Rabinowitz (ed.), *Numerical Methods for Nonlinear Algebraic Equations*,
/// Gordon and Breach, 1970, 87–114.
pub fn solve_system<S, F>(f: F, initial: &[S]) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    if initial.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("initial"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "system dimension must be at least one",
        ));
    }
    let function = ResidualFn::new(initial.len(), f);
    let result =
        TrustRegionSystem::new(TrustRegionSystemOptions::default())?.solve(&function, initial)?;
    crate::termination::require_success(result.termination(), result.accounting())?;
    Ok(result.root.into_vec())
}
