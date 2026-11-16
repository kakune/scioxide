//! Execution options shared by linear-algebra operations.

use scioxide_core::ExecutionPolicy;

/// Execution policy, product accuracy and resource limits for a linear-algebra call.
///
/// The defaults are [`ExecutionPolicy::Auto`],
/// [`Accuracy::Standard`](scioxide_core::Accuracy::Standard), and no time, byte or
/// iteration limit. Dense operations and factorizations
/// take these options directly. Solver and plan options embed them through
/// their `with_execution_options` method. Mathematical choices such as
/// pivoting or the stored triangle are separate arguments, not options here.
///
/// Owned forms always run sequentially. The `*_into` forms apply the execution
/// policy to the [`LinalgExecutionContext`](crate::LinalgExecutionContext) they
/// receive. See [Execution](crate#execution).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_linalg::LinalgExecutionOptions;
/// let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct LinalgExecutionOptions {
    execution: ExecutionPolicy,
    accuracy: scioxide_core::Accuracy,
    time_limit: Option<core::time::Duration>,
    byte_limit: Option<usize>,
    iterations: Option<scioxide_core::Budget>,
}

impl LinalgExecutionOptions {
    /// Returns the default options: `Auto` execution, `Standard` accuracy and no limits.
    pub const fn new() -> Self {
        Self {
            execution: ExecutionPolicy::Auto,
            accuracy: scioxide_core::Accuracy::Standard,
            time_limit: None,
            byte_limit: None,
            iterations: None,
        }
    }

    /// Sets how many workers a `*_into` call may use.
    ///
    /// The policy only states a limit. The workers themselves come from the
    /// execution context of the call.
    pub const fn with_execution_policy(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the accuracy of dense products, dot products and norms.
    ///
    /// See [`Accuracy`](scioxide_core::Accuracy) for the error bound of each level.
    pub const fn with_accuracy(mut self, accuracy: scioxide_core::Accuracy) -> Self {
        self.accuracy = accuracy;
        self
    }

    /// Returns the product accuracy, `Standard` by default.
    pub const fn accuracy(self) -> scioxide_core::Accuracy {
        self.accuracy
    }

    /// Returns the execution policy, `Auto` by default.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution
    }

    /// Sets a limit on elapsed time, measured with the clock of the execution context.
    ///
    /// A call whose context has no clock, including every owned form, returns
    /// `CapabilityUnavailable` before any work. A failing clock or a deadline
    /// that overflows is `CapabilityFailure`.
    pub const fn with_time_limit(mut self, duration: core::time::Duration) -> Self {
        self.time_limit = Some(duration);
        self
    }

    /// Returns the time limit, `None` by default.
    pub const fn time_limit(self) -> Option<core::time::Duration> {
        self.time_limit
    }

    /// Sets a limit in bytes on workspace storage.
    ///
    /// The limit covers all workspace storage, including capacity kept from
    /// earlier calls. It excludes fixed-size metadata and the output you
    /// provide. Plans that store numerical data, such as structured solves and
    /// sparse materializations, count that data too. Owned structured
    /// materializations and Khatri–Rao products apply the limit to the result
    /// they allocate. A requirement over the limit is `ResourceLimit`, reported
    /// before any allocation.
    pub const fn with_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = Some(bytes);
        self
    }

    /// Returns the workspace byte limit, `None` by default.
    pub const fn byte_limit(self) -> Option<usize> {
        self.byte_limit
    }

    /// Sets the iteration limit of iterative algorithms, such as SVD sweeps or LDL pivot blocks.
    ///
    /// What counts as one iteration depends on the algorithm. `k = min(rows, cols)`.
    ///
    /// - SVD: each QR sweep and each secular-equation root step counts as one.
    ///   The default is `30 k` for QR iteration and
    ///   `30 k (1 + ⌈log₂ ⌈k/32⌉⌉)` for divide and conquer.
    /// - GSVD and CSD: preparation counts as one and each Jacobi sweep as one.
    ///   The default allows 40 sweeps.
    /// - QR updates: each elementary update counts as one.
    /// - Polar decomposition: preparation counts as one.
    /// - Inner SVD stages of these decompositions get the same limit and
    ///   count in their own scopes.
    /// - Riccati equations: preparation and recovery count toward the limit.
    ///   The QZ, reordering and SVD stages get the same limit. Without one, QZ
    ///   and reordering default to 30,000.
    /// - Iterative linear solvers run at most the smaller of this limit and
    ///   their own `max_iterations`.
    /// - Iterative refinement makes at most this many corrections per
    ///   right-hand side, 10 by default.
    /// - Partial spectrum and partial SVD: each restart cycle counts as one.
    ///   LOBPCG: each outer iteration counts as one. All three default to 100.
    /// - Exponential actions: each Taylor term or power-norm product counts as
    ///   one. Function actions: each Arnoldi step. One-norm estimates: each
    ///   operator application. Randomized low-rank plans: each sketch column or
    ///   power step. These have no default limit.
    /// - An action counts an iteration only if its operator-application limit
    ///   still allows that step's application. Projected dense stages have
    ///   their own inner limits.
    ///
    /// Kernels without iteration, such as products, count nothing. Iterations
    /// of a nested solve count toward the nested scope, not the parent's
    /// iteration limit (see [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn with_iteration_budget(mut self, iterations: scioxide_core::Budget) -> Self {
        self.iterations = Some(iterations);
        self
    }

    /// Returns the iteration limit, `None` by default.
    ///
    /// Each algorithm documents the limit it uses when this is `None`.
    pub const fn iteration_budget(self) -> Option<scioxide_core::Budget> {
        self.iterations
    }

    /// Returns `limit` capped by the iteration limit, if one is set.
    pub(crate) fn cap_iterations(self, limit: scioxide_core::Budget) -> scioxide_core::Budget {
        self.iterations
            .map_or(limit, |allowance| allowance.min(limit))
    }

    pub(crate) fn solve_budgets(self) -> scioxide_core::SolveBudgets {
        let budgets = scioxide_core::SolveBudgets::unbounded();
        match self.iterations {
            Some(limit) => budgets.with_limit(scioxide_core::BudgetKind::Iteration, limit),
            None => budgets,
        }
    }
}

impl Default for LinalgExecutionOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Choice of whether a plan computes an optional diagnostic estimate.
///
/// Examples are condition numbers after Schur reordering, closed-loop
/// stability of a Riccati solution, the separation of a Sylvester equation and
/// the error of a sparse materialization. These estimates cost extra work and
/// scratch. With [`Skip`](Self::Skip) the corresponding report field is absent.
/// Each options type documents its default.
///
/// ```
/// use scioxide_linalg::Estimation;
/// assert_ne!(Estimation::Skip, Estimation::Estimate);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimation {
    /// Do not compute the estimate.
    Skip,
    /// Compute the estimate and report it.
    Estimate,
}
