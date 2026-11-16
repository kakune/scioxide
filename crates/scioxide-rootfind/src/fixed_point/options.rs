//! Options for fixed-point solves.
use super::*;
/// Selects how [`FixedPointSolver`] reads the output of the residual operator.
///
/// In both cases the solver works with the residual `f(x)`, and the damped
/// Picard step is `x + β f(x)` with damping `β` from
/// [`FixedPointOptions::with_damping`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPointInput {
    /// The callback returns `G(x)` for the fixed-point problem `x = G(x)`; the
    /// residual is `f(x) = G(x) − x`.
    Map,
    /// The callback returns `F(x)` for the root problem `F(x) = 0`; the residual is
    /// `f(x) = F(x)`, so the damped step `x + β F(x)` is linear mixing.
    Residual,
}

/// Selects the iteration or acceleration method of [`FixedPointSolver`].
///
/// All methods start from the damped Picard step `x + β f(x)`, where `f` is the
/// residual defined by [`FixedPointInput`] and `β` is the damping. History methods
/// store at most `min(history, n)` pairs of steps `Δx` and residual differences
/// `Δf` for `n` unknowns. For complex scalars, inner products are Hermitian.
///
/// # References
///
/// - H. F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations",
///   *SIAM J. Numer. Anal.* 49(4), 2011.
/// - C. G. Broyden, "A class of methods for solving nonlinear simultaneous
///   equations", *Math. Comp.* 19(92), 1965.
/// - C. T. Kelley, *Iterative Methods for Linear and Nonlinear Equations*, SIAM,
///   1995, §7.2 (Broyden's method).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPointMethod {
    /// Damped Picard iteration `x ← x + β f(x)`. One residual evaluation per
    /// iteration.
    Picard,
    /// Steffensen's method: Aitken Δ² extrapolation of each coordinate of the
    /// damped step. Two residual evaluations per iteration. A coordinate whose
    /// denominator is close to zero keeps the Picard value and counts as a
    /// skipped update.
    Steffensen,
    /// Type-II Anderson acceleration. Each iteration solves a least-squares
    /// problem over the stored `Δf` columns, after the component divisors, by
    /// column-pivoted Householder QR with the configured rank policy. Without
    /// regularization, columns beyond the numerical rank are dropped from the
    /// history; with Tikhonov regularization all columns are kept.
    Anderson,
    /// Broyden's first ("good") method, applied as a rank-one update of an
    /// approximate inverse Jacobian `H`, starting from `H₀ = −β I`.
    BroydenFirst,
    /// Broyden's second ("bad") method, updating the approximate inverse Jacobian
    /// directly, starting from `H₀ = −β I`.
    BroydenSecond,
    /// Broyden's first update restricted to a diagonal Jacobian approximation. When
    /// an updated diagonal entry nearly vanishes, its step coefficient is reset to
    /// `β` and the reset is counted in [`FixedPointDiagnostics::restarts`].
    DiagonalBroyden,
    /// Diagonal mixing with one coefficient per component, for real scalars only.
    /// A coefficient grows by `β` while its residual component keeps its sign, up to
    /// [`FixedPointOptions::with_maximum_mixing`], and resets to `β` when the sign
    /// changes.
    ExcitingMixing,
}

/// Selects what a history method does when its history is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryRestart {
    /// Drop the oldest pair and keep the most recent ones.
    Oldest,
    /// Drop all pairs before storing the new one. Each restart is counted in
    /// [`FixedPointDiagnostics::restarts`].
    Restart,
}

/// Configures a [`FixedPointSolver`].
///
/// The solve succeeds with [`RootTermination::ResidualTolerance`] when the
/// residual norm `‖D⁻¹ f(x)‖₂` passes the residual tolerance, where `D` holds the
/// component divisors (default 1). The relative part of the tolerance is measured
/// against the norm at the initial point, or against
/// [`with_residual_scale`](Self::with_residual_scale). A step with
/// `‖Δx‖₂ ≤ atol + rtol · max(1, ‖x‖₂)` ends the solve with
/// [`RootTermination::Stagnation`] only if the residual did not decrease; a zero
/// step tolerance disables this test.
///
/// The step test does not certify a root; only the residual test does.
///
/// | Control | Default |
/// |---|---|
/// | damping `β` | 1 |
/// | history | 5 pairs, [`HistoryRestart::Oldest`] |
/// | residual tolerance | absolute 0, relative `√ε` |
/// | step tolerance | absolute and relative `√ε` |
/// | iterations | 200 |
/// | maximum exciting-mixing coefficient | 1 (raised to `β` by `with_damping`) |
/// | Anderson rank policy | [`RankPolicy::DimensionScaledEpsilon`], no regularization |
///
/// [`FixedPointSolver::new`] validates these controls.
#[derive(Clone, Debug)]
pub struct FixedPointOptions<R: RealScalar> {
    pub(crate) damping: R,
    pub(crate) history: usize,
    pub(crate) regularization: R,
    pub(crate) rank: RankPolicy<R>,
    pub(crate) restart: HistoryRestart,
    pub(crate) residual: ResidualTolerance<R>,
    pub(crate) step: StepTolerance<R>,
    pub(crate) scale: Option<R>,
    pub(crate) component_scales: Vec<R>,
    pub(crate) iterations: Budget,
    pub(crate) maximum_mixing: R,
    pub(crate) budgets: scioxide_core::SolveBudgets,
    pub(crate) policy: ExecutionPolicy,
    pub(crate) byte_limit: usize,
}
impl<R: RealScalar> Default for FixedPointOptions<R> {
    fn default() -> Self {
        Self {
            damping: R::ONE,
            history: 5,
            regularization: R::ZERO,
            rank: RankPolicy::DimensionScaledEpsilon,
            restart: HistoryRestart::Oldest,
            residual: crate::termination::relative_tolerance(),
            step: StepTolerance::default(),
            scale: None,
            component_scales: Vec::new(),
            iterations: Budget::from(core::num::NonZeroUsize::MIN).saturating_add(199),
            maximum_mixing: R::ONE,
            budgets: scioxide_core::SolveBudgets::unbounded(),
            policy: ExecutionPolicy::Sequential,
            byte_limit: usize::MAX,
        }
    }
}
impl<R: RealScalar> FixedPointOptions<R> {
    /// Sets the work budgets of each solve. The limits of an enclosing solve also
    /// apply (see [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn with_budgets(mut self, value: scioxide_core::SolveBudgets) -> Self {
        self.budgets = value;
        self
    }
    /// Returns the work budgets of each solve.
    pub const fn budgets(&self) -> scioxide_core::SolveBudgets {
        self.budgets
    }
    /// Sets the execution policy of the Anderson QR factorization (default
    /// sequential).
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.policy = value;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Sets the largest workspace size in bytes (default unlimited).
    pub const fn with_workspace_byte_limit(mut self, value: usize) -> Self {
        self.byte_limit = value;
        self
    }
    /// Returns the largest workspace size in bytes.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        positive(self.damping, scioxide_core::argument!("options.damping"))?;
        positive(
            self.maximum_mixing,
            scioxide_core::argument!("options.maximum_mixing"),
        )?;
        if self.maximum_mixing < self.damping {
            return Err(invalid(
                scioxide_core::argument!("options.maximum_mixing"),
                InvalidInputReason::Inconsistent,
                "maximum mixing is below damping",
            ));
        }
        if self.history == 0 {
            return Err(invalid(
                scioxide_core::argument!("options.history"),
                InvalidInputReason::Inconsistent,
                "history capacity must be positive",
            ));
        }
        if !self.regularization.is_finite() || self.regularization < R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.regularization"),
                InvalidInputReason::Inconsistent,
                "regularization must be finite and nonnegative",
            ));
        }
        if let Some(scale) = self.scale {
            positive(scale, scioxide_core::argument!("options.residual_scale"))?;
        }
        self.component_scales
            .iter()
            .try_for_each(|&v| positive(v, scioxide_core::argument!("options.component_scales")))
    }

    /// Sets the damping `β`, which must be positive and finite. It also raises the
    /// maximum exciting-mixing coefficient to at least `β`.
    pub fn with_damping(mut self, damping: R) -> Self {
        self.damping = damping;
        self.maximum_mixing = self.maximum_mixing.max(damping);
        self
    }
    /// Sets the number of stored history pairs, which must be positive, and the
    /// rule applied when the history is full. The workspace stores at most
    /// `min(capacity, n)` pairs.
    pub fn with_history(mut self, capacity: usize, restart: HistoryRestart) -> Self {
        self.history = capacity;
        self.restart = restart;
        self
    }
    /// Sets the Tikhonov parameter of the Anderson least-squares problem, which
    /// must be finite and nonnegative (default 0).
    pub fn with_regularization(mut self, regularization: R) -> Self {
        self.regularization = regularization;
        self
    }
    /// Sets the numerical-rank policy of the Anderson QR factorization.
    pub fn with_rank_policy(mut self, rank: RankPolicy<R>) -> Self {
        self.rank = rank;
        self
    }
    /// Sets the residual tolerance, which decides success (default absolute 0,
    /// relative `√ε`).
    pub fn with_residual_tolerance(mut self, residual: ResidualTolerance<R>) -> Self {
        self.residual = residual;
        self
    }
    /// Sets the step tolerance, which detects stagnation (default absolute and
    /// relative `√ε`). A zero tolerance disables the test.
    pub fn with_step_tolerance(mut self, step: StepTolerance<R>) -> Self {
        self.step = step;
        self
    }
    /// Sets the reference norm of the relative residual tolerance, in place of the
    /// initial residual norm. It must be positive and finite.
    pub fn with_residual_scale(mut self, scale: R) -> Self {
        self.scale = Some(scale);
        self
    }
    /// Sets positive divisors `D` for the residual components, used in the stopping
    /// test and the Anderson least-squares problem. An empty vector means unit
    /// divisors; otherwise its length must equal the problem dimension.
    pub fn with_component_scales(mut self, scales: Vec<R>) -> Self {
        self.component_scales = scales;
        self
    }
    /// Sets the iteration limit (default 200). The work budgets may stop the solve
    /// earlier.
    pub fn with_max_iterations(mut self, iterations: Budget) -> Self {
        self.iterations = iterations;
        self
    }
    /// Sets the upper bound of the [`FixedPointMethod::ExcitingMixing`]
    /// coefficients. It must be positive, finite and at least the damping.
    pub fn with_maximum_mixing(mut self, maximum: R) -> Self {
        self.maximum_mixing = maximum;
        self
    }
    pub(crate) fn divisor(&self, index: usize) -> R {
        self.component_scales.get(index).copied().unwrap_or(R::ONE)
    }
}
