//! Two-point boundary value problems by collocation with residual control.
//!
//! [`BvpPlan::solve_into`] solves into a [`BvpSolution`] you provide and keeps its
//! storage in a [`BvpWorkspace`]; [`BvpPlan::solve`] returns an owned solution.
use crate::safety::finite;
use crate::safety::invalid;
use crate::shell::Shell;
use crate::termination::{Fault, Work, admit, record};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, try_copy_slice, try_filled};
use scioxide_core::{
    Budget, BudgetKind, EvaluationCounts, EvaluationRole, ExecutionContext, NonFiniteOrigin,
    NumericScalar, PartialResultTrust, ProblemToken, RealScalar, ResidualTolerance, SciOxideError,
    SolveBudgets, SolveContext, SolveCounts, SolveIdentity, SolveScope, StepTolerance, Tolerance,
};
use scioxide_linalg::{Matrix, MatrixView, RankPolicy};
use scioxide_rootfind::{
    NewtonKrylovForcing, NewtonKrylovOptions, NewtonKrylovSolver, RootTermination,
};
use scioxide_sparse::SparseLuLimits;

mod assembly;
mod collocation;
mod linear;
mod solution;
mod workspace;

use collocation::Collocation;
use linear::DirectCorrection;
pub use solution::BvpSolution;
use workspace::Storage;
pub use workspace::{BvpRequirements, BvpWorkspace};

fn r<R: RealScalar>(x: f64) -> Result<R, SciOxideError> {
    R::try_from_f64_lossy(x)
}
/// Defines a two-point boundary value problem for [`BvpPlan`].
///
/// ```text
/// y′(x) = f(x, y, p),   a ≤ x ≤ b,
/// g(y(a), y(b), p) = 0,
/// ```
///
/// where `y` has `n` = [`Self::dimension`] components, `p` holds `k` =
/// [`Self::parameter_dimension`] unknown parameters, `f` is [`Self::rhs`] and `g` is
/// [`Self::boundary`], which has `n + k` components. The interval `[a, b]` is the span
/// of the plan's mesh. With [`BvpPlan::with_singular`], the solver adds the term
/// `S y/(x − a)` to `f`.
///
/// `x` is real. For complex states and parameters, `f` and `g` must be holomorphic,
/// because the Jacobians are complex-linear. The callbacks must write every output
/// component and must not change their data or dimensions during a solve. Components
/// left unwritten, NaN and infinity are rejected as `NonFinite` with origin
/// `Callback`. The order of the calls is unspecified.
///
/// Each call counts as one evaluation in the solve's [accounting](scioxide_core#solve-accounting):
/// `RightHandSide` for [`Self::rhs`] and `Constraint` for [`Self::boundary`]. Charge
/// only the nested work of your callback, such as an inner solve, to `cx`.
pub trait BvpSystem<T: NumericScalar> {
    /// Returns the number of state components `n`, which must be positive.
    fn dimension(&self) -> usize;
    /// Returns the number of unknown parameters `k`; the default is zero.
    fn parameter_dimension(&self) -> usize {
        0
    }
    /// Evaluates `f(x, y, p)` into `output`, without the singular term.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the solve returns unchanged.
    fn rhs(
        &self,
        x: T::Real,
        y: &[T],
        p: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates the `n + k` boundary residuals `g(y(a), y(b), p)` into `output`.
    ///
    /// `left` is `y(a)` and `right` is `y(b)`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the solve returns unchanged.
    fn boundary(
        &self,
        left: &[T],
        right: &[T],
        p: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Supplies analytic Jacobians of a [`BvpSystem`]; use it with [`BvpProblem::analytic`].
///
/// All matrices are row-major. The right-hand side blocks `∂f/∂y` and `∂f/∂p` have
/// shapes `n × n` and `n × k`. The boundary blocks `∂g/∂y(a)`, `∂g/∂y(b)` and `∂g/∂p`
/// have shapes `(n + k) × n`, `(n + k) × n` and `(n + k) × k`. Blocks for `k = 0` are
/// empty. Write every entry, including zeros. Each call counts as one `Jacobian`
/// evaluation; charge only nested work to `cx`.
pub trait BvpJacobian<T: NumericScalar>: BvpSystem<T> {
    /// Evaluates `∂f/∂y` into `state` and `∂f/∂p` into `parameters` at `(x, y, p)`.
    ///
    /// The singular term of [`BvpPlan::with_singular`] is added by the solver.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation unchanged. Non-finite entries are a
    /// `NonFinite` error with origin `Callback`.
    fn rhs_jacobian(
        &self,
        x: T::Real,
        y: &[T],
        p: &[T],
        state: &mut [T],
        parameters: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates `∂g/∂y(a)` into `a`, `∂g/∂y(b)` into `b` and `∂g/∂p` into `parameters`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation unchanged. Non-finite entries are a
    /// `NonFinite` error with origin `Callback`.
    #[allow(clippy::too_many_arguments)]
    fn boundary_jacobian(
        &self,
        left: &[T],
        right: &[T],
        p: &[T],
        a: &mut [T],
        b: &mut [T],
        parameters: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

type RhsJacobian<T, P> = fn(
    &P,
    <T as NumericScalar>::Real,
    &[T],
    &[T],
    &mut [T],
    &mut [T],
    &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>;
type BoundaryJacobian<T, P> = fn(
    &P,
    &[T],
    &[T],
    &[T],
    &mut [T],
    &mut [T],
    &mut [T],
    &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>;
struct JacobianCallbacks<T: NumericScalar, P: ?Sized> {
    rhs: RhsJacobian<T, P>,
    boundary: BoundaryJacobian<T, P>,
}
impl<T: NumericScalar, P: ?Sized> Copy for JacobianCallbacks<T, P> {}
impl<T: NumericScalar, P: ?Sized> Clone for JacobianCallbacks<T, P> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Pairs a borrowed [`BvpSystem`] with a choice of Jacobian.
///
/// [`Self::numerical`] approximates the Jacobian by forward differences of the
/// callbacks; [`Self::analytic`] calls [`BvpJacobian`].
pub struct BvpProblem<'a, T: NumericScalar, P: BvpSystem<T> + ?Sized> {
    system: &'a P,
    jacobian: Option<JacobianCallbacks<T, P>>,
}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Copy for BvpProblem<'_, T, P> {}
impl<T: NumericScalar, P: BvpSystem<T> + ?Sized> Clone for BvpProblem<'_, T, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T: NumericScalar, P: BvpSystem<T> + ?Sized> BvpProblem<'a, T, P> {
    /// Borrows `system` and approximates Jacobians by forward differences.
    ///
    /// The direct backends difference each local block of `f` and `g`. With
    /// [`BvpBackend::Krylov`], the solver differences Jacobian–vector products of the
    /// whole collocation system instead.
    pub fn numerical(system: &'a P) -> Self {
        Self {
            system,
            jacobian: None,
        }
    }
    /// Borrows `system` and evaluates Jacobians with its [`BvpJacobian`] methods.
    pub fn analytic(system: &'a P) -> Self
    where
        P: BvpJacobian<T>,
    {
        Self {
            system,
            jacobian: Some(JacobianCallbacks {
                rhs: P::rhs_jacobian,
                boundary: P::boundary_jacobian,
            }),
        }
    }
}

/// Selects how [`BvpPlan`] solves the linear system of each Newton step.
///
/// The solver uses the selected method only; it never switches to another one. The
/// default is [`Self::Sparse`] with limits derived from the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpBackend {
    /// Assembles the dense Jacobian and solves by LU with partial pivoting.
    ///
    /// The Jacobian has `(m n + k)²` entries for `m` mesh nodes, so use it only for
    /// small systems.
    Dense,
    /// Assembles the block-sparse Jacobian and solves by sparse LU with pivoting.
    ///
    /// Each interval contributes `n` rows that couple two adjacent nodes, and the
    /// boundary rows couple both endpoints, so storage grows linearly in `m`.
    Sparse {
        /// Fill and memory limits of the sparse factorization. The byte limit of
        /// [`BvpOptions::new`] still applies. `None` derives the limits from each mesh:
        /// four times the number of entries of the block pattern, and a quarter of the
        /// workspace byte limit.
        limits: Option<SparseLuLimits>,
    },
    /// Solves by restarted GMRES without assembling the Jacobian.
    ///
    /// Jacobian–vector products use the analytic local Jacobians of
    /// [`BvpProblem::analytic`], or finite differences of the collocation residual for
    /// [`BvpProblem::numerical`].
    Krylov {
        /// Largest number of GMRES iterations per Newton step.
        iterations: Budget,
        /// Number of Krylov basis vectors kept before a restart.
        restart: Budget,
    },
}
impl Default for BvpBackend {
    /// Returns [`Self::Sparse`] with limits derived from each mesh.
    fn default() -> Self {
        Self::Sparse { limits: None }
    }
}

/// Options of [`BvpPlan`]: mesh and memory limits, tolerances, refinements, Newton
/// controls and the linear backend.
///
/// The setters do not check their arguments; [`BvpPlan::new`] validates the whole
/// configuration.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BvpOptions<R: RealScalar> {
    max_nodes: usize,
    max_bytes: usize,
    tolerances: (ResidualTolerance<R>, ResidualTolerance<R>),
    refinements: Budget,
    newton: BvpNewtonOptions<R>,
    backend: BvpBackend,
}

/// Controls the damped Newton iteration that solves the collocation equations on each
/// mesh.
///
/// The iteration uses a backtracking line search and, with [`BvpBackend::Krylov`],
/// inexact Newton steps with forcing terms. Its stopping test follows the tolerances
/// of [`BvpOptions`]. The defaults are those of [`NewtonKrylovOptions`]: 100
/// iterations, Armijo constant `10⁻⁴`, smallest relative step `ε^(2/3)`, where `ε` is
/// the machine epsilon, nonmonotone weight 0.85 and Eisenstat–Walker forcing.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BvpNewtonOptions<R: RealScalar> {
    max_iterations: Budget,
    armijo: R,
    minimum_step: R,
    nonmonotone: R,
    forcing: NewtonKrylovForcing<R>,
}
impl<R: RealScalar> Default for BvpNewtonOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> BvpNewtonOptions<R> {
    /// Returns the default controls.
    pub fn new() -> Self {
        let sixteen = R::TWO.powi(4);
        Self {
            max_iterations: Budget::ONE.saturating_add(99),
            armijo: (R::ONE / R::TEN).powi(4),
            minimum_step: R::CBRT_EPSILON * R::CBRT_EPSILON,
            nonmonotone: (sixteen + R::ONE) / (sixteen + R::TWO + R::TWO),
            forcing: NewtonKrylovForcing::default(),
        }
    }
    /// Sets the largest number of Newton iterations on each mesh; linear iterations do
    /// not count.
    pub fn with_max_iterations(mut self, iterations: Budget) -> Self {
        self.max_iterations = iterations;
        self
    }
    /// Sets the Armijo constant `c ∈ (0, 1)` and the smallest relative step
    /// `minimum ∈ (0, 1]` of the line search; see
    /// [`NewtonKrylovOptions::with_backtracking`].
    pub fn with_backtracking(mut self, c: R, minimum: R) -> Self {
        self.armijo = c;
        self.minimum_step = minimum;
        self
    }
    /// Sets the nonmonotone line-search weight `w ∈ [0, 1]`; zero gives the monotone
    /// Armijo test.
    ///
    /// See [`NewtonKrylovOptions::with_nonmonotone`].
    pub fn with_nonmonotone(mut self, weight: R) -> Self {
        self.nonmonotone = weight;
        self
    }
    /// Sets the forcing terms of [`BvpBackend::Krylov`] steps; the direct backends solve
    /// each Newton step exactly.
    pub fn with_forcing(mut self, forcing: NewtonKrylovForcing<R>) -> Self {
        self.forcing = forcing;
        self
    }
    /// Returns the Newton iteration limit on each mesh.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
    /// Returns the Armijo constant `c` and the smallest relative step `minimum` of the
    /// line search.
    pub const fn backtracking(&self) -> (R, R) {
        (self.armijo, self.minimum_step)
    }
    /// Returns the nonmonotone line-search weight `w`.
    pub const fn nonmonotone(&self) -> R {
        self.nonmonotone
    }
    /// Returns the forcing terms of Krylov steps.
    pub const fn forcing(&self) -> NewtonKrylovForcing<R> {
        self.forcing
    }
    /// Converts these settings to Newton–Krylov solver options.
    fn solver_options(&self) -> NewtonKrylovOptions<R> {
        NewtonKrylovOptions::default()
            .with_max_iterations(self.max_iterations)
            .with_backtracking(self.armijo, self.minimum_step)
            .with_nonmonotone(self.nonmonotone)
            .with_forcing(self.forcing)
    }
}

impl<R: RealScalar> BvpOptions<R> {
    /// Creates options with no node or workspace byte limit.
    /// Limits are compared, not reserved; storage follows the current mesh.
    ///
    /// [`BvpPlan::new`] requires at least two nodes and a positive byte limit. The byte
    /// count is an upper bound on the workspace storage that the plan computes from the
    /// mesh size and dimensions. The other defaults are residual tolerances
    /// `atol = rtol = 10⁻³`, boundary tolerance `atol = 10⁻³`, [`BvpBackend::Sparse`]
    /// with limits derived from each mesh, at most 20 refinements and
    /// [`BvpNewtonOptions::new`].
    pub fn new() -> Self {
        let (max_nodes, max_workspace_bytes) = (usize::MAX, usize::MAX);
        Self {
            max_nodes,
            max_bytes: max_workspace_bytes,
            // `10^(-u32::MAX)` underflows to the zero relative boundary tolerance.
            tolerances: (
                ResidualTolerance::from_tolerance(Tolerance::from_decimal_places(3, 3)),
                ResidualTolerance::from_tolerance(Tolerance::from_decimal_places(3, u32::MAX)),
            ),
            refinements: Budget::ONE.saturating_add(19),
            newton: BvpNewtonOptions::new(),
            backend: BvpBackend::default(),
        }
    }
    /// Limits mesh nodes without reserving storage; the default has no limit.
    pub fn with_max_nodes(mut self, nodes: usize) -> Self {
        self.max_nodes = nodes;
        self
    }
    /// Limits the workspace bound, including twice the derived sparse LU allowance.
    /// The default has no limit; the bound is compared, not reserved.
    pub fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Sets the tolerances of the collocation residual and of the boundary residual.
    ///
    /// The residual of the cubic solution `u` is `r(x) = u′(x) − f(x, u(x), p)`. Each
    /// component is divided by `atol + rtol |fᵢ(x, u(x), p)|`. On every interval
    /// `[xⱼ, xⱼ₊₁]` of length `hⱼ`, the solve requires
    ///
    /// ```text
    /// ( (1/hⱼ) ∫ ‖r̃(x)‖₂² dx )^(1/2) ≤ 1,
    /// ```
    ///
    /// where `r̃` is the scaled residual and the integral is evaluated by five-point
    /// Lobatto quadrature. The boundary residual must satisfy `maxᵢ |gᵢ| ≤ atol`. Both
    /// absolute tolerances must be positive and the boundary relative tolerance must be
    /// zero.
    pub fn with_tolerances(
        mut self,
        residual: ResidualTolerance<R>,
        boundary: ResidualTolerance<R>,
    ) -> Self {
        self.tolerances = (residual, boundary);
        self
    }
    /// Sets the linear solver of the Newton steps.
    pub fn with_backend(mut self, backend: BvpBackend) -> Self {
        self.backend = backend;
        self
    }
    /// Sets the largest number of mesh refinements.
    pub fn with_refinements(mut self, refinements: Budget) -> Self {
        self.refinements = refinements;
        self
    }
    /// Sets the Newton iteration limit, line search and Krylov forcing terms.
    pub fn with_newton(mut self, options: BvpNewtonOptions<R>) -> Self {
        self.newton = options;
        self
    }
    /// Returns the largest permitted number of mesh nodes.
    pub const fn max_nodes(&self) -> usize {
        self.max_nodes
    }
    /// Returns the limit on the workspace storage bound, in bytes.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_bytes
    }
    /// Returns the residual and boundary tolerances.
    pub const fn tolerances(&self) -> (ResidualTolerance<R>, ResidualTolerance<R>) {
        self.tolerances
    }
    /// Returns the largest number of mesh refinements.
    pub const fn refinements(&self) -> Budget {
        self.refinements
    }
    /// Returns the Newton controls.
    pub const fn newton(&self) -> &BvpNewtonOptions<R> {
        &self.newton
    }
    /// Returns the linear solver of the Newton steps.
    pub const fn backend(&self) -> BvpBackend {
        self.backend
    }
}

/// Dimensions and linear algorithm that fix the collocation storage layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    n: usize,
    k: usize,
    backend: BvpBackend,
}
impl Layout {
    fn sparse_limits(self, nodes: usize, bytes: usize) -> Result<SparseLuLimits, SciOxideError> {
        if let BvpBackend::Sparse {
            limits: Some(limits),
        } = self.backend
        {
            return Ok(limits);
        }
        let width = checked_add(
            checked_mul(2, self.n, scioxide_core::quantity!("sparse_width"))?,
            self.k,
            scioxide_core::quantity!("sparse_width"),
        )?;
        let entries = checked_mul(
            checked_mul(
                self.size(nodes)?,
                width,
                scioxide_core::quantity!("sparse_entries"),
            )?,
            4,
            scioxide_core::quantity!("sparse_entries"),
        )?;
        Ok(SparseLuLimits::new(entries, bytes / 4))
    }
    fn size(self, nodes: usize) -> Result<usize, SciOxideError> {
        checked_add(
            checked_mul(
                nodes,
                self.n,
                scioxide_core::quantity!("collocation_values"),
            )?,
            self.k,
            scioxide_core::quantity!("collocation_unknowns"),
        )
    }
}

/// Checks the workspace storage bound of `nodes` against `limit`.
fn storage_bound<T: NumericScalar>(
    layout: Layout,
    nodes: usize,
    limit: usize,
    argument: scioxide_core::Argument,
) -> Result<usize, SciOxideError> {
    let bytes = storage_bytes::<T>(layout, nodes, limit)?;
    if bytes > limit {
        Err(invalid(
            argument,
            "BVP conservative workspace bound exceeds caller byte limit",
        ))
    } else {
        Ok(bytes)
    }
}

// Conservative retained-capacity bound, including refinement staging,
// nonlinear/differentiation storage, indices and owned solution publication.
fn storage_bytes<T: NumericScalar>(
    layout: Layout,
    nodes: usize,
    limit: usize,
) -> Result<usize, SciOxideError> {
    let dim = layout.size(nodes)?;
    let base = checked_mul(dim, 1024, scioxide_core::quantity!("workspace_bytes"))?;
    let blocks = checked_mul(
        checked_add(
            layout.n,
            layout.k,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        checked_add(
            layout.n,
            layout.k,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let extra = match layout.backend {
        BvpBackend::Dense => checked_mul(
            checked_mul(dim, dim, scioxide_core::quantity!("workspace_bytes"))?,
            32,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        BvpBackend::Sparse { .. } => checked_mul(
            checked_mul(
                dim,
                checked_add(
                    checked_mul(layout.n, 2, scioxide_core::quantity!("workspace_bytes"))?,
                    layout.k,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            8,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        // Krylov basis and Hessenberg storage, and the cached local Jacobians of
        // analytic products.
        BvpBackend::Krylov { restart, .. } => checked_add(
            checked_mul(
                checked_add(
                    checked_mul(
                        dim,
                        restart.get(),
                        scioxide_core::quantity!("workspace_bytes"),
                    )?,
                    checked_mul(
                        restart.get(),
                        restart.get(),
                        scioxide_core::quantity!("workspace_bytes"),
                    )?,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                8,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            collocation::jacobian_len(layout.n, layout.k, nodes)?,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
    };
    let scalars = checked_add(
        checked_add(base, extra, scioxide_core::quantity!("workspace_bytes"))?,
        checked_mul(blocks, 128, scioxide_core::quantity!("workspace_bytes"))?,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let bytes = checked_mul(
        scalars,
        core::mem::size_of::<T>().max(core::mem::size_of::<usize>()),
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    if matches!(layout.backend, BvpBackend::Sparse { .. }) {
        let limits = layout.sparse_limits(nodes, limit)?;
        checked_add(
            bytes,
            checked_mul(
                limits.max_workspace_bytes(),
                2,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )
    } else {
        Ok(bytes)
    }
}

#[derive(Debug, Clone)]
struct Singular<T: NumericScalar> {
    matrix: Matrix<T>,
    projection: Matrix<T>,
    derivative: Matrix<T>,
}

/// Boundary and singular-regularity residuals of one assessed mesh.
struct Assessment<R> {
    boundary: R,
    regularity: R,
    bound_bytes: usize,
}

/// Solves two-point boundary value problems by collocation with C¹ cubic splines.
///
/// Construction copies the initial mesh in `O(m)` work for `m` nodes.
/// [`Self::with_singular`] precomputes two pseudo-inverses with SVDs of order `n`,
/// costing `O(n³)` work and `O(n²)` storage for `n` state components.
///
/// The plan holds the initial mesh `a = x₀ < x₁ < … < xₘ₋₁ = b`, the dimensions `n`
/// and `k` of a [`BvpSystem`], and the [`BvpOptions`]. On a mesh, the unknowns are
/// the node values `yⱼ` and the parameters `p`. The solve enforces the boundary
/// conditions `g(y₀, yₘ₋₁, p) = 0` and, on every interval of length `hⱼ`, the
/// three-stage Lobatto IIIA (Simpson) collocation equations
///
/// ```text
/// yⱼ₊½ = (yⱼ + yⱼ₊₁)/2 − hⱼ (fⱼ₊₁ − fⱼ)/8,
/// (yⱼ₊₁ − yⱼ)/hⱼ = (fⱼ + 4 fⱼ₊½ + fⱼ₊₁)/6,
/// ```
///
/// where `fⱼ = f(xⱼ, yⱼ, p)` and `fⱼ₊½` is `f` at the interval midpoint and `yⱼ₊½`.
/// The solution is the C¹ cubic Hermite interpolant of `yⱼ` and `fⱼ`, fourth-order
/// accurate at the nodes.
///
/// A damped Newton iteration with a backtracking line search solves these
/// `m n + k` equations; [`BvpBackend`] selects the linear solver. The plan then
/// estimates the residual `u′ − f(x, u, p)` of the cubic `u` on each interval (see
/// [`BvpOptions::with_tolerances`]), bisects every interval whose scaled residual
/// exceeds one, and solves again on the refined mesh, starting from `u`. Each
/// residual evaluation calls [`BvpSystem::rhs`] `4m − 3` times and
/// [`BvpSystem::boundary`] once.
///
/// The residual is controlled, not the error `u − y`. The two are close for a
/// well-conditioned problem, but the reported residuals are not error bounds.
///
/// A plan validates its options and mesh once, is immutable, and can solve any
/// problem with the same dimensions; see
/// [Reusing workspaces](crate#reusing-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError, SolveContext};
/// use scioxide_linalg::Matrix;
/// use scioxide_ode::{BvpOptions, BvpPlan, BvpProblem, BvpSolution, BvpSystem, BvpTermination};
/// // y″ = 0 with y(0) = 0 and y(1) = 1, as the first-order system (y, y′).
/// struct Line;
/// impl BvpSystem<f64> for Line {
///     fn dimension(&self) -> usize {
///         2
///     }
///     fn rhs(
///         &self,
///         _: f64,
///         y: &[f64],
///         _: &[f64],
///         out: &mut [f64],
///         _: &mut SolveContext<'_, '_>,
///     ) -> Result<(), SciOxideError> {
///         out.copy_from_slice(&[y[1], 0.0]);
///         Ok(())
///     }
///     fn boundary(
///         &self,
///         a: &[f64],
///         b: &[f64],
///         _: &[f64],
///         out: &mut [f64],
///         _: &mut SolveContext<'_, '_>,
///     ) -> Result<(), SciOxideError> {
///         out.copy_from_slice(&[a[0], b[0] - 1.0]);
///         Ok(())
///     }
/// }
/// let plan = BvpPlan::new(
///     Line.dimension(),
///     Line.parameter_dimension(),
///     &[0.0, 0.5, 1.0],
///     BvpOptions::new().with_max_nodes(64).with_max_workspace_bytes(1 << 24),
/// )?;
/// let problem = BvpProblem::numerical(&Line);
/// let mut workspace = plan.workspace(&problem)?;
/// let mut solution = BvpSolution::new();
/// let guess = Matrix::zeros(3, 2)?;
/// let report = plan.solve_into(
///     &problem,
///     guess.view(),
///     &[],
///     &mut solution,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(report.termination(), BvpTermination::Converged);
/// // The exact solution is y(x) = x.
/// let mut y = [0.0; 2];
/// solution.evaluate_into(0.25, &mut y)?;
/// assert!((y[0] - 0.25).abs() < 1e-8);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Kierzenka and L. F. Shampine, *ACM Trans. Math. Softw.* 27(3), 299–316, 2001
///   (Lobatto IIIA collocation with residual control).
/// - U. M. Ascher, R. M. M. Mattheij and R. D. Russell, *Numerical Solution of
///   Boundary Value Problems for Ordinary Differential Equations*, SIAM, 1995.
/// - L. F. Shampine, "Singular boundary value problems for ODEs", *Appl. Math.
///   Comput.* 138, 99–112, 2003.
#[derive(Debug, Clone)]
#[must_use]
pub struct BvpPlan<T: NumericScalar> {
    options: BvpOptions<T::Real>,
    mesh: Vec<T::Real>,
    layout: Layout,
    residual: ResidualTolerance<T::Real>,
    boundary: ResidualTolerance<T::Real>,
    max_nodes: usize,
    max_bytes: usize,
    refinements: Budget,
    newton: NewtonKrylovSolver<T>,
    singular: Option<Singular<T>>,
}
impl<T: NumericScalar> BvpPlan<T> {
    /// Creates a plan for `dimension` states and `parameter_dimension` unknown
    /// parameters on the initial `mesh`.
    ///
    /// The plan has no singular term; add one with [`Self::with_singular`]. No callback
    /// runs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options` allow fewer than two nodes or zero bytes, if an
    ///   absolute tolerance is not positive or the boundary relative tolerance is not
    ///   zero, or if the Newton options are invalid.
    /// - `InvalidInput` if `dimension` is zero, or if `mesh` has fewer than two or more
    ///   than [`BvpOptions::max_nodes`] points, is not strictly increasing, or has a
    ///   non-finite point or span.
    /// - `InvalidInput` if the workspace storage bound of the initial mesh exceeds
    ///   [`BvpOptions::max_workspace_bytes`].
    /// - `SizeOverflow` or `AllocationFailure` if the plan cannot be stored.
    pub fn new(
        dimension: usize,
        parameter_dimension: usize,
        mesh: &[T::Real],
        options: BvpOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let BvpOptions {
            max_nodes,
            max_bytes,
            tolerances,
            refinements,
            ref newton,
            backend,
        } = options;
        if max_nodes < 2 || max_bytes == 0 {
            return Err(invalid(
                if max_nodes < 2 {
                    scioxide_core::argument!("options.max_nodes")
                } else {
                    scioxide_core::argument!("options.max_workspace_bytes")
                },
                "positive storage and at least two nodes are required",
            ));
        }
        let (residual, boundary) = tolerances;
        if residual.absolute() <= T::Real::ZERO
            || boundary.absolute() <= T::Real::ZERO
            || boundary.relative() != T::Real::ZERO
        {
            return Err(invalid(
                if residual.absolute() <= T::Real::ZERO {
                    scioxide_core::argument!("options.residual")
                } else {
                    scioxide_core::argument!("options.boundary")
                },
                "positive absolute tolerances and absolute-only boundary tolerance required",
            ));
        }
        if dimension == 0
            || mesh.len() < 2
            || mesh.len() > max_nodes
            || mesh.iter().any(|x| !x.is_finite())
            || !(mesh[mesh.len() - 1] - mesh[0]).is_finite()
            || mesh
                .windows(2)
                .any(|x| x[0] >= x[1] || !(x[1] - x[0]).is_finite())
        {
            return Err(invalid(
                if dimension == 0 {
                    scioxide_core::argument!("system")
                } else {
                    scioxide_core::argument!("mesh")
                },
                "positive dimension and finite strictly increasing bounded mesh required",
            ));
        }
        // Unitless thresholds: the scaling provider supplies physical tolerances,
        // including precision floors, separately for every equation and variable.
        let mut newton = newton
            .solver_options()
            .with_residual_tolerance(ResidualTolerance::new(T::Real::ONE, T::Real::ZERO)?)
            .with_step_tolerance(StepTolerance::new(
                T::Real::ZERO,
                (residual.relative() * r(0.001)?).max(T::Real::EPSILON * r(32.)?),
            )?);
        if let BvpBackend::Krylov {
            iterations,
            restart,
        } = backend
        {
            newton = newton.with_krylov(iterations, restart);
        }
        let mut plan = Self {
            options,
            mesh: Vec::new(),
            layout: Layout {
                n: dimension,
                k: parameter_dimension,
                backend,
            },
            residual,
            boundary,
            max_nodes,
            max_bytes,
            refinements,
            newton: NewtonKrylovSolver::new(newton)?,
            singular: None,
        };
        plan.storage_bound(mesh.len(), scioxide_core::argument!("options"))?;
        plan.mesh = try_copy_slice(mesh, scioxide_core::quantity!("mesh"))?;
        Ok(plan)
    }
    /// Returns the number of states `n`.
    pub const fn dimension(&self) -> usize {
        self.layout.n
    }
    /// Returns the number of unknown parameters `k`.
    pub const fn parameter_dimension(&self) -> usize {
        self.layout.k
    }
    /// Returns the options the plan was created with.
    pub const fn options(&self) -> &BvpOptions<T::Real> {
        &self.options
    }
    /// Adds the singular term `S y/(x − a)` to the right-hand side.
    ///
    /// The differential equation becomes `y′ = S y/(x − a) + f(x, y, p)` with a
    /// constant `n × n` matrix `S`. The solve assumes a C¹ solution with `S y(a) = 0`
    /// and boundary conditions consistent with it. At `x = a` the solver uses
    /// `y′(a) = (I − S)⁺ f(a, y(a), p)`, where `⁺` is the pseudoinverse with rank
    /// determined at a dimension-scaled multiple of `ε`. Each solve also projects
    /// `y(a)` onto the null space of `S`, and [`BvpReport::regularity_residual`]
    /// reports `max |S y(a)|`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `matrix` is not `n × n`, or if the workspace storage bound
    ///   of the initial mesh exceeds the byte limit.
    /// - `NonFinite` (origin `Input`) if `matrix` has a NaN or infinite entry.
    /// - The errors of the singular value decompositions of `S` and `I − S`, and
    ///   `AllocationFailure`.
    ///
    /// A solve fails with `InvalidInput` if `f(a, y(a), p)` is not in the range of
    /// `I − S` to within the boundary tolerance.
    ///
    /// # References
    ///
    /// L. F. Shampine, "Singular boundary value problems for ODEs", *Appl. Math.
    /// Comput.* 138, 99–112, 2003.
    pub fn with_singular(mut self, matrix: MatrixView<'_, T>) -> Result<Self, SciOxideError> {
        let n = self.layout.n;
        if matrix.rows() != n || matrix.cols() != n {
            return Err(invalid(
                scioxide_core::argument!("matrix"),
                "singular matrix must be n by n",
            ));
        }
        self.storage_bound(self.mesh.len(), scioxide_core::argument!("self"))?;
        let mut s = Matrix::zeros(n, n)?;
        for i in 0..n {
            for j in 0..n {
                s.as_mut_slice()[i * n + j] = matrix.get(i, j)?;
            }
        }
        finite(s.as_slice(), NonFiniteOrigin::Input)?;
        let inverse = scioxide_linalg::SvdDecomposition::compute(&s)?
            .pseudoinverse(RankPolicy::DimensionScaledEpsilon)?;
        let product = inverse.view().mul_mat(&s)?;
        let mut projection = Matrix::identity(n)?;
        let mut effective = Matrix::identity(n)?;
        projection
            .as_mut_slice()
            .iter_mut()
            .zip(product.as_slice())
            .for_each(|(a, b)| *a = *a - *b);
        effective
            .as_mut_slice()
            .iter_mut()
            .zip(s.as_slice())
            .for_each(|(a, b)| *a = *a - *b);
        let derivative = scioxide_linalg::SvdDecomposition::compute(&effective)?
            .pseudoinverse(RankPolicy::DimensionScaledEpsilon)?;
        self.singular = Some(Singular {
            matrix: s,
            projection,
            derivative,
        });
        Ok(self)
    }
    fn storage_bound(
        &self,
        nodes: usize,
        argument: scioxide_core::Argument,
    ) -> Result<usize, SciOxideError> {
        storage_bound::<T>(self.layout, nodes, self.max_bytes, argument)
    }
    fn check_problem<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
    ) -> Result<(), SciOxideError> {
        if problem.system.dimension() != self.layout.n
            || problem.system.parameter_dimension() != self.layout.k
        {
            Err(invalid(
                scioxide_core::argument!("problem"),
                "problem dimensions differ from prepared plan",
            ))
        } else {
            Ok(())
        }
    }
    /// Returns the workspace storage needed to solve `problem` on the initial mesh.
    ///
    /// Mesh refinement grows the workspace during a solve, within the node and byte
    /// limits of the options. The workspace keeps the grown storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the dimensions of `problem` differ from the plan's, or if
    ///   the storage bound exceeds the byte limit.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
    ) -> Result<BvpRequirements<T>, SciOxideError> {
        self.check_problem(problem)?;
        self.requirements(problem, &self.mesh)
    }
    /// Allocates a workspace for `problem` on the initial mesh; no callback runs.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
    ) -> Result<BvpWorkspace<T>, SciOxideError> {
        let mut workspace = BvpWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(workspace)
    }
    fn preflight<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        guess: MatrixView<'_, T>,
        parameters: &[T],
        w: &mut Storage<T>,
    ) -> Result<(), SciOxideError> {
        self.check_problem(problem)?;
        let Layout { n, k, .. } = self.layout;
        let m = self.mesh.len();
        if guess.rows() != m || guess.cols() != n || parameters.len() != k {
            return Err(invalid(
                if guess.rows() != m || guess.cols() != n {
                    scioxide_core::argument!("guess")
                } else {
                    scioxide_core::argument!("parameters")
                },
                "initial state or parameter shape differs from plan",
            ));
        }
        w.reserve(&self.requirements(problem, &self.mesh)?)?;
        resize(&mut w.mesh, m, T::Real::ZERO)?;
        w.mesh.copy_from_slice(&self.mesh);
        for i in 0..m {
            for j in 0..n {
                w.point[i * n + j] = guess.get(i, j)?;
            }
        }
        w.point[m * n..].copy_from_slice(parameters);
        finite(&w.point, NonFiniteOrigin::Input)
    }
    /// Solves `problem` from an initial guess and writes the solution to `out`.
    ///
    /// `guess` holds the initial node values, one row per node of the initial mesh and
    /// one column per state; strided views are accepted. `parameters` is the initial
    /// guess for `p`. Every solve starts from the initial mesh of the plan. The solve
    /// evaluates the callbacks sequentially and runs in `scope`, which is an
    /// [`ExecutionContext`] or the `&mut SolveContext` of an enclosing solve; see
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// The method is described on [`BvpPlan`]. The solve returns `Ok` with the report
    /// whenever it has a solution on some mesh, including unsuccessful terminations
    /// such as [`BvpTermination::NodeLimit`]; check [`BvpReport::is_success`]. The
    /// report is also stored in `out`. On error, `out` is unchanged, and
    /// [`BvpWorkspace::last_attempt`] records the counts of the failed attempt. With a
    /// workspace that has already solved on the final mesh and an `out` of sufficient
    /// capacity, the solve does not allocate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the dimensions of `problem` differ from the plan's, if
    ///   `guess` is not `m × n` for the `m` initial nodes, or if `parameters` does not
    ///   have length `k`. No callback runs.
    /// - `NonFinite` (origin `Input`) if `guess` or `parameters` has a NaN or infinite
    ///   value. No callback runs.
    /// - `NonFinite` (origin `Callback`) if a callback returns NaN or infinity, or
    ///   leaves an output component unwritten; `NonFinite` (origin `Intermediate`) if
    ///   an iterate or scaled residual overflows.
    /// - `InvalidInput` if a singular term is set and `f(a, y(a), p)` is incompatible
    ///   with it (see [`Self::with_singular`]).
    /// - `ConvergenceFailure` if an iteration or evaluation budget of `scope` is
    ///   exhausted, and `ExecutionStopped` on cancellation or a deadline.
    /// - Errors returned by the callbacks, unchanged, and `AllocationFailure`.
    pub fn solve_into<'scope, 'capability, P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        guess: MatrixView<'_, T>,
        parameters: &[T],
        out: &mut BvpSolution<T>,
        workspace: &mut BvpWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<BvpReport<T::Real>, SciOxideError> {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, guess, parameters, &mut workspace.storage) {
            return workspace.shell.reject(error);
        }
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "bvp-collocation", 1),
            self.layout.n as u128,
            0,
        );
        let BvpWorkspace { storage, shell } = workspace;
        let mut executed = shell.execute(scope, identity, SolveBudgets::unbounded(), |cx| {
            self.collocate(problem, storage, cx)
        });
        if let Ok(report) = executed.result
            && let Err(error) = storage.publish(self.layout.n, report, out)
        {
            executed.result = Err(error);
        }
        let report = executed.result.as_ref().ok().copied();
        shell.conclude(executed, BvpReport::is_success, |_| report)
    }
    /// Solves `problem` from an initial guess and returns the solution.
    ///
    /// This is [`Self::solve_into`] with a new workspace and
    /// [`ExecutionContext::sequential_native`]. The report is in
    /// [`BvpSolution::report`]; the solution is returned also when the termination is
    /// unsuccessful.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::solve_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{SciOxideError, SolveContext};
    /// use scioxide_linalg::Matrix;
    /// use scioxide_ode::{BvpOptions, BvpPlan, BvpProblem, BvpSystem, BvpTermination};
    /// // y′ = 2, y(0) = 0; the exact solution is y(x) = 2x.
    /// struct Linear;
    /// impl BvpSystem<f64> for Linear {
    ///     fn dimension(&self) -> usize {
    ///         1
    ///     }
    ///     fn rhs(
    ///         &self,
    ///         _: f64,
    ///         _: &[f64],
    ///         _: &[f64],
    ///         out: &mut [f64],
    ///         _: &mut SolveContext<'_, '_>,
    ///     ) -> Result<(), SciOxideError> {
    ///         out[0] = 2.0;
    ///         Ok(())
    ///     }
    ///     fn boundary(
    ///         &self,
    ///         left: &[f64],
    ///         _: &[f64],
    ///         _: &[f64],
    ///         out: &mut [f64],
    ///         _: &mut SolveContext<'_, '_>,
    ///     ) -> Result<(), SciOxideError> {
    ///         out[0] = left[0];
    ///         Ok(())
    ///     }
    /// }
    /// let options = BvpOptions::new().with_max_nodes(16).with_max_workspace_bytes(1_048_576);
    /// let plan = BvpPlan::new(1, 0, &[0.0, 1.0], options)?;
    /// let guess = Matrix::zeros(2, 1)?;
    /// let solution = plan.solve(&BvpProblem::numerical(&Linear), guess.view(), &[])?;
    /// let report = solution.report().expect("a solve stores its report");
    /// assert_eq!(report.termination(), BvpTermination::Converged);
    /// let mut value = [0.0];
    /// solution.evaluate_into(0.5, &mut value)?;
    /// assert!((value[0] - 1.0).abs() < 1e-6);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    pub fn solve<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        guess: MatrixView<'_, T>,
        parameters: &[T],
    ) -> Result<BvpSolution<T>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut solution = BvpSolution::new();
        // The published solution carries the report.
        let _ = self.solve_into(
            problem,
            guess,
            parameters,
            &mut solution,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(solution)
    }
    fn collocate<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        w: &mut Storage<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<BvpReport<T::Real>, SciOxideError> {
        let n = self.layout.n;
        if let Some(s) = &self.singular {
            mat_apply(&s.projection, &w.point[..n], &mut w.projected);
            w.point[..n].copy_from_slice(&w.projected);
            finite(&w.point, NonFiniteOrigin::Intermediate)?;
        }
        let mut refinements = 0;
        loop {
            record(context, Work::Iteration)?;
            w.reserve(&self.requirements(problem, &w.mesh)?)?;
            let collocation = Collocation {
                plan: self,
                problem,
                mesh: &w.mesh,
                cache: Some(&w.evaluations),
                direct: w.direct.as_ref(),
            };
            let correction = w.direct.as_ref().map(|direct| direct.bind(&collocation));
            let child_problem = collocation.nonlinear_problem(
                correction
                    .as_ref()
                    .map(|direct| direct as &dyn scioxide_rootfind::NonlinearLinearSolver<T>),
            );
            let result = w.result.as_mut().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "nonlinear result storage is not reserved",
                )
            })?;
            scioxide_rootfind::backend::solve_newton_partitioned(
                &self.newton,
                &child_problem,
                &w.point,
                result,
                &mut w.newton,
                &mut w.nonlinear_scope,
                &mut *context,
            )?;
            w.point.copy_from_slice(result.point());
            let nonlinear = (!result.is_success()).then(|| result.termination());
            if let Some(singular) = &self.singular {
                mat_apply(&singular.projection, &w.point[..n], &mut w.projected);
                w.point[..n].copy_from_slice(&w.projected);
                finite(&w.point, NonFiniteOrigin::Intermediate)?;
            }
            let assessed = self.assess(problem, w, context)?;
            let solution = &w.assessment;
            let mut report = BvpReport {
                termination: BvpTermination::Converged,
                refinements,
                nodes: w.mesh.len(),
                max_residual: solution.max_residual(),
                boundary_residual: assessed.boundary,
                regularity_residual: assessed.regularity,
                counts: context.counts(),
                workspace_bytes: assessed.bound_bytes,
            };
            if let Some(termination) = nonlinear {
                report.termination = BvpTermination::Nonlinear(termination);
                return Ok(report);
            }
            if report.max_residual <= T::Real::ONE
                && report.boundary_residual <= self.boundary.absolute()
                && report.regularity_residual <= self.boundary.absolute()
            {
                context.commit_accepted(PartialResultTrust::ResidualChecked)?;
                report.counts = context.counts();
                return Ok(report);
            }
            if refinements >= self.refinements.get() {
                report.termination = BvpTermination::RefinementLimit;
                return Ok(report);
            }
            let count = solution
                .residuals
                .iter()
                .filter(|x| **x > T::Real::ONE)
                .count();
            let force_first = report.regularity_residual > self.boundary.absolute();
            let count = count + usize::from(force_first && solution.residuals[0] <= T::Real::ONE);
            if count == 0 {
                report.termination = BvpTermination::BoundaryFailure;
                return Ok(report);
            }
            let nodes = checked_add(w.mesh.len(), count, scioxide_core::quantity!("mesh_nodes"))?;
            if nodes > self.max_nodes {
                report.termination = BvpTermination::NodeLimit;
                return Ok(report);
            }
            // An unrepresentable bound also exceeds every byte limit.
            if !storage_bytes::<T>(self.layout, nodes, self.max_bytes)
                .is_ok_and(|bytes| bytes <= self.max_bytes)
            {
                report.termination = BvpTermination::MemoryLimit;
                return Ok(report);
            }
            record(context, Work::Proposal)?;
            match record(context, Work::Subdivisions(count)) {
                Ok(()) => {}
                Err(Fault::Denied(denial)) => {
                    report.termination = BvpTermination::ResourceLimit(denial.kind());
                    report.counts = context.counts();
                    return Ok(report);
                }
                Err(Fault::Error(error)) => return Err(error),
            }
            resize(&mut w.next_mesh, w.mesh.len() + count, T::Real::ZERO)?;
            let mut next_index = 0;
            for (i, interval) in w.mesh.windows(2).enumerate() {
                w.next_mesh[next_index] = interval[0];
                next_index += 1;
                if solution.residuals[i] > T::Real::ONE || (i == 0 && force_first) {
                    let middle = interval[0] + (interval[1] - interval[0]) * T::Real::HALF;
                    if middle <= interval[0] || middle >= interval[1] {
                        report.termination = BvpTermination::MeshStagnation;
                        return Ok(report);
                    }
                    w.next_mesh[next_index] = middle;
                    next_index += 1;
                }
            }
            w.next_mesh[next_index] = w.mesh[w.mesh.len() - 1];
            resize(
                &mut w.next_point,
                self.layout.size(w.next_mesh.len())?,
                T::ZERO,
            )?;
            for (i, &x) in w.next_mesh.iter().enumerate() {
                solution.evaluate_into(x, &mut w.next_point[i * n..(i + 1) * n])?;
            }
            w.next_point[w.next_mesh.len() * n..].copy_from_slice(solution.parameters());
            core::mem::swap(&mut w.mesh, &mut w.next_mesh);
            core::mem::swap(&mut w.point, &mut w.next_point);
            refinements += 1;
        }
    }
    fn assess<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        w: &mut Storage<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Assessment<T::Real>, SciOxideError> {
        let n = self.layout.n;
        let m = w.mesh.len();
        let solution = &mut w.assessment;
        let collocation = Collocation {
            plan: self,
            problem,
            mesh: &w.mesh,
            cache: Some(&w.evaluations),
            direct: None,
        };
        let scratch = &mut w.assessment_scratch[..n];
        collocation.ensure_evaluated(&w.point, &mut w.interpolation, context)?;
        let cache = collocation.cached()?;
        solution.derivatives.copy_from_slice(&cache.rhs);
        let bc = &mut w.boundary;
        bc.copy_from_slice(&cache.residual[(m - 1) * n..]);
        let boundary = crate::safety::max_abs(bc.iter().copied());
        let regularity = if let Some(s) = &self.singular {
            mat_apply(&s.matrix, &w.point[..n], scratch);
            finite(scratch, NonFiniteOrigin::Intermediate)?;
            crate::safety::max_abs(scratch.iter().copied())
        } else {
            T::Real::ZERO
        };
        if !boundary.is_finite() || !regularity.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "boundary or regularity residual norm overflow",
            ));
        }
        solution.mesh.copy_from_slice(&w.mesh);
        solution.values.copy_from_slice(&w.point[..m * n]);
        solution.parameters.copy_from_slice(&w.point[m * n..]);
        let bound_bytes = self.storage_bound(w.capacity_nodes, scioxide_core::argument!("self"))?;
        let offset = r::<T::Real>(3.0 / 7.0)?.sqrt() * T::Real::HALF;
        let (value, work) = w.interpolation.split_at_mut(n);
        let (derivative, rhs) = work.split_at_mut(n);
        for i in 0..m - 1 {
            let h = w.mesh[i + 1] - w.mesh[i];
            let mut sum = T::Real::ZERO;
            for (q, (fraction, weight)) in [
                (T::Real::HALF, r::<T::Real>(16.0 / 45.0)?),
                (T::Real::HALF - offset, r(49.0 / 180.0)?),
                (T::Real::HALF + offset, r(49.0 / 180.0)?),
            ]
            .into_iter()
            .enumerate()
            {
                hermite(
                    h,
                    fraction,
                    &w.point[i * n..(i + 1) * n],
                    &w.point[(i + 1) * n..(i + 2) * n],
                    &cache.rhs[i * n..(i + 1) * n],
                    &cache.rhs[(i + 1) * n..(i + 2) * n],
                    value,
                    Some(derivative),
                );
                finite(value, NonFiniteOrigin::Intermediate)?;
                finite(derivative, NonFiniteOrigin::Intermediate)?;
                rhs.copy_from_slice(if q == 0 {
                    &cache.middle_rhs[i * n..(i + 1) * n]
                } else {
                    &cache.quadrature_rhs[(2 * i + q - 1) * n..(2 * i + q) * n]
                });
                let norm = derivative.iter().zip(rhs.iter()).try_fold(
                    T::Real::ZERO,
                    |sum, (&d, &f)| {
                        let denominator =
                            self.residual.absolute() + self.residual.relative() * f.abs();
                        if !denominator.is_finite() {
                            return Err(SciOxideError::non_finite(
                                NonFiniteOrigin::Intermediate,
                                "residual normalization denominator overflow",
                            ));
                        }
                        let v = (d - f).abs() / denominator;
                        let out = sum + v * v;
                        if out.is_finite() {
                            Ok(out)
                        } else {
                            Err(SciOxideError::non_finite(
                                NonFiniteOrigin::Intermediate,
                                "residual normalization overflow",
                            ))
                        }
                    },
                )?;
                sum += weight * norm;
            }
            solution.residuals[i] = sum.sqrt();
        }
        Ok(Assessment {
            boundary,
            regularity,
            bound_bytes,
        })
    }
}

fn resize<V: Clone>(values: &mut Vec<V>, len: usize, value: V) -> Result<(), SciOxideError> {
    let additional = len.saturating_sub(values.len());
    scioxide_core::backend::try_reserve(
        values,
        additional,
        scioxide_core::quantity!("bvp_storage"),
    )?;
    values.resize(len, value);
    Ok(())
}

/// Reports why a [`BvpPlan`] solve stopped.
///
/// Only [`Self::Converged`] meets the tolerances. For every other value, the solution
/// of the last mesh is still returned, together with its residuals, so you can inspect
/// it or use it as the guess of a new solve. A budget, cancellation or deadline that
/// stops a solve before a mesh is assessed is an error, not a termination; see
/// [`BvpPlan::solve_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BvpTermination {
    /// The residual, boundary and regularity tolerances are met.
    Converged,
    /// The refined mesh would have more nodes than [`BvpOptions::max_nodes`].
    NodeLimit,
    /// The workspace storage bound of the refined mesh would exceed
    /// [`BvpOptions::max_workspace_bytes`].
    MemoryLimit,
    /// The solve performed [`BvpOptions::refinements`] refinements without meeting
    /// the tolerances.
    RefinementLimit,
    /// A budget of the enclosing scope does not allow the next refinement.
    ResourceLimit(BudgetKind),
    /// An interval to be bisected has no representable midpoint.
    MeshStagnation,
    /// The boundary residual exceeds its tolerance although every interval meets the
    /// residual tolerance, so refinement cannot help.
    BoundaryFailure,
    /// The Newton iteration on the current mesh stopped without converging, for the
    /// given reason.
    Nonlinear(RootTermination),
}
impl BvpTermination {
    /// Returns whether the termination is [`Self::Converged`].
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Converged)
    }
}

/// Reports the termination, final mesh, residuals and work of a [`BvpPlan`] solve.
///
/// The residuals are computed from the final cubic solution. They measure how well it
/// satisfies the equations, not its distance to the exact solution.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BvpReport<R: RealScalar> {
    termination: BvpTermination,
    refinements: usize,
    nodes: usize,
    max_residual: R,
    boundary_residual: R,
    regularity_residual: R,
    counts: SolveCounts,
    workspace_bytes: usize,
}
impl<R: RealScalar> BvpReport<R> {
    /// Returns why the solve stopped.
    pub const fn termination(&self) -> BvpTermination {
        self.termination
    }
    /// Returns whether the solve met every tolerance.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the number of completed mesh refinements.
    pub const fn refinements(&self) -> usize {
        self.refinements
    }
    /// Returns the number of nodes of the final mesh.
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    /// Returns the largest scaled interval residual (see
    /// [`BvpOptions::with_tolerances`]); convergence requires at most one.
    pub const fn max_residual(&self) -> R {
        self.max_residual
    }
    /// Returns `maxᵢ |gᵢ|` of the boundary residual.
    pub const fn boundary_residual(&self) -> R {
        self.boundary_residual
    }
    /// Returns `max |S y(a)|` for the singular term of [`BvpPlan::with_singular`]; zero
    /// without one.
    pub const fn regularity_residual(&self) -> R {
        self.regularity_residual
    }
    /// Returns the evaluations and iterations of the collocation solve itself,
    /// excluding nested solves.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the counts of the solve together with its nested solves.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the storage bound of the workspace at the end of the solve, in bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

fn mat_apply<T: NumericScalar>(a: &Matrix<T>, x: &[T], out: &mut [T]) {
    for (i, value) in out.iter_mut().enumerate() {
        *value = a.as_slice()[i * a.cols()..(i + 1) * a.cols()]
            .iter()
            .zip(x)
            .fold(T::ZERO, |sum, (&a, &x)| sum + a * x);
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn hermite<T: NumericScalar>(
    h: T::Real,
    t: T::Real,
    a: &[T],
    b: &[T],
    fa: &[T],
    fb: &[T],
    out: &mut [T],
    mut derivative: Option<&mut [T]>,
) {
    let one = T::Real::ONE;
    let two = one + one;
    let three = two + one;
    let h00 = (one + two * t) * (one - t) * (one - t);
    let h10 = t * (one - t) * (one - t);
    let h01 = t * t * (three - two * t);
    let h11 = t * t * (t - one);
    for (i, value) in out.iter_mut().enumerate() {
        *value = a[i] * T::from_real(h00)
            + b[i] * T::from_real(h01)
            + fa[i] * T::from_real(h * h10)
            + fb[i] * T::from_real(h * h11);
    }
    if let Some(d) = derivative.as_mut() {
        for (i, value) in d.iter_mut().enumerate() {
            *value = (b[i] - a[i]) * T::from_real((two * three * t * (one - t)) / h)
                + fa[i] * T::from_real(one - two * two * t + three * t * t)
                + fb[i] * T::from_real(three * t * t - two * t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_defaults_are_the_shared_newton_krylov_defaults() {
        assert_eq!(
            BvpNewtonOptions::<f64>::new().solver_options(),
            NewtonKrylovOptions::default()
        );
        assert_eq!(
            BvpNewtonOptions::<f32>::new().solver_options(),
            NewtonKrylovOptions::default()
        );
    }
}

impl<R: RealScalar> Default for BvpOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Solves `y′ = rhs(x, y)` with `boundary(y(a), y(b)) = 0` on `mesh`.
///
/// `guess` is node-major, with one row per mesh node and one column per state.
/// Uses fourth-order collocation and adaptive residual control, with residual
/// absolute and relative tolerances `1e-3`. Storage follows the refined mesh and
/// sparse factorization. Returns the cubic solution with its convergence report.
///
/// # Errors
/// Returns `InvalidInput` for an empty state, the errors of [`BvpPlan::new`] and
/// [`BvpPlan::solve`], and convergence or resource errors for unsuccessful stops.
///
/// # Examples
/// ```
/// let guess = scioxide_linalg::Matrix::zeros(3, 2)?;
/// let solution = scioxide_ode::solve_bvp(
///     |_x: f64, y: &[f64], dy: &mut [f64]| { dy[0]=y[1]; dy[1]=-y[0]; Ok(()) },
///     |a: &[f64], b: &[f64], r: &mut [f64]| { r[0]=a[0]; r[1]=b[0]-1.0; Ok(()) },
///     &[0.0, 0.5, 1.0], guess.view(),
/// )?;
/// let mut y = [0.0; 2];
/// solution.evaluate_into(0.5, &mut y)?;
/// assert!((y[0] - 0.5_f64.sin() / 1.0_f64.sin()).abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Kierzenka and Shampine (2001), *A BVP solver based on residual control*.
pub fn solve_bvp<T, F, G>(
    rhs: F,
    boundary: G,
    mesh: &[T::Real],
    guess: MatrixView<'_, T>,
) -> Result<BvpSolution<T>, SciOxideError>
where
    T: NumericScalar,
    F: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    if guess.cols() == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("guess"),
            scioxide_core::InvalidInputReason::Empty,
            "guess must have a state column",
        ));
    }
    let system = crate::BvpFn::new(guess.cols(), rhs, boundary);
    let plan = BvpPlan::new(guess.cols(), 0, mesh, BvpOptions::default())?;
    let solution = plan.solve(&BvpProblem::numerical(&system), guess, &[])?;
    if let Some(report) = solution.report()
        && let Some(error) = termination_error(&report)
    {
        return Err(error);
    }
    Ok(solution)
}

fn termination_error<R: RealScalar>(report: &BvpReport<R>) -> Option<SciOxideError> {
    use scioxide_core::{ConvergenceDiagnostics, ConvergenceReason};
    let reason = match report.termination() {
        BvpTermination::Converged => return None,
        BvpTermination::RefinementLimit => ConvergenceReason::IterationLimit,
        BvpTermination::MeshStagnation | BvpTermination::BoundaryFailure => {
            ConvergenceReason::Stagnation
        }
        BvpTermination::Nonlinear(_) => ConvergenceReason::NestedSolver,
        BvpTermination::ResourceLimit(k) => ConvergenceReason::ResourceLimit(k),
        BvpTermination::NodeLimit => {
            return Some(SciOxideError::resource_limit(
                scioxide_core::quantity!("mesh_nodes"),
                report.nodes().saturating_add(1),
                usize::MAX,
            ));
        }
        BvpTermination::MemoryLimit => {
            return Some(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                report.workspace_bytes(),
                usize::MAX,
            ));
        }
    };
    Some(SciOxideError::convergence_failure(
        reason,
        ConvergenceDiagnostics::new().with_counts(report.accounting()),
        "boundary-value solve did not converge",
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod one_shot_tests {
    use super::*;
    use scioxide_core::{ConvergenceReason, SciOxideErrorView};
    fn report(termination: BvpTermination) -> BvpReport<f64> {
        BvpReport {
            termination,
            refinements: 1,
            nodes: 3,
            max_residual: 0.1,
            boundary_residual: 0.2,
            regularity_residual: 0.,
            counts: SolveCounts::new(),
            workspace_bytes: 1234,
        }
    }
    #[test]
    fn every_unsuccessful_termination_is_an_error() {
        assert!(termination_error(&report(BvpTermination::Converged)).is_none());
        for (termination, expected) in [
            (
                BvpTermination::RefinementLimit,
                ConvergenceReason::IterationLimit,
            ),
            (
                BvpTermination::MeshStagnation,
                ConvergenceReason::Stagnation,
            ),
            (
                BvpTermination::BoundaryFailure,
                ConvergenceReason::Stagnation,
            ),
            (
                BvpTermination::Nonlinear(RootTermination::IterationLimit),
                ConvergenceReason::NestedSolver,
            ),
            (
                BvpTermination::ResourceLimit(BudgetKind::Subdivision),
                ConvergenceReason::ResourceLimit(BudgetKind::Subdivision),
            ),
        ] {
            let report = report(termination);
            let error = termination_error(&report).unwrap();
            assert!(
                matches!(error.view(),SciOxideErrorView::ConvergenceFailure{reason,diagnostics,..} if reason==expected && diagnostics.counts()==report.accounting())
            );
        }
        for (termination, expected) in [
            (
                BvpTermination::NodeLimit,
                scioxide_core::quantity!("mesh_nodes"),
            ),
            (
                BvpTermination::MemoryLimit,
                scioxide_core::quantity!("workspace_bytes"),
            ),
        ] {
            let error = termination_error(&report(termination)).unwrap();
            assert!(
                matches!(error.view(),SciOxideErrorView::ResourceLimit{resource,..} if resource==expected)
            );
        }
    }
}
