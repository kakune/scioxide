//! Bounded FAQ relaxation and permutation 2-opt for real quadratic assignment.
//!
//! Prepared FAQ uses two block GEMMs for the asymmetric gradient, caches seed
//! interactions across restarts and evaluates the line-search quadratic from
//! those products and an assignment endpoint. Sequential prepared calls do not
//! allocate; numerical failure leaves the output result unchanged.
use alloc::vec::Vec;
use scioxide_core::Argument;
use scioxide_core::backend::{
    PartitionSlot, PartitionedSolve, RealProductSum, checked_add, checked_mul, try_copy_slice,
    try_filled, try_with_capacity, with_partitioned_solves,
};
use scioxide_core::{
    AttemptKind, Budget, ConvergenceReason, ExecutionContext, ExecutionPolicy, ExecutionStopReason,
    NonFiniteOrigin, NumericScalar, PermutationView, ProblemToken, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveScope, StepTolerance,
};
use scioxide_linalg::{
    DenseWorkspaceRequest, LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace, MatrixView, MatrixViewMut,
};
use scioxide_random::{SeedableRng, Xoshiro256StarStar, uniform_index};
use scioxide_sparse::{
    AssignmentObjective, AssignmentOutcome, AssignmentPlan, AssignmentWorkspace,
    GraphExecutionOptions, SparseView,
};

use crate::OptimizeTermination;
use crate::validation::invalid;

const FAQ_KERNELS: LinalgExecutionOptions =
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);

/// Both rectangular products share an arena sized for their maximum requirements.
fn faq_requirements<S: RealScalar>(k: usize) -> Result<LinalgRequirements, SciOxideError> {
    let twice = checked_mul(2, k, scioxide_core::quantity!("faq_block_order"))?;
    let first = LinalgWorkspace::<S>::workspace_requirements(&DenseWorkspaceRequest::product(
        twice,
        k,
        k,
        &FAQ_KERNELS,
    ))?;
    let second = LinalgWorkspace::<S>::workspace_requirements(&DenseWorkspaceRequest::product(
        k,
        twice,
        k,
        &FAQ_KERNELS,
    ))?;
    Ok(first.max(&second))
}

/// Evaluate the line-search quadratic, preserving its endpoint values.
fn line_value<S: RealScalar>(
    f0: S,
    f1: S,
    derivative: S,
    curvature: S,
    step: S,
) -> Result<S, SciOxideError> {
    if step == S::ZERO {
        return Ok(f0);
    }
    if step == S::ONE {
        return Ok(f1);
    }
    let mut value = RealProductSum::<S>::new();
    value.add_product(f0, S::ONE)?;
    value.add_product(derivative, step)?;
    value.add_product(finite(curvature * step)?, step)?;
    value.finish()
}

fn finite<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "unrepresentable quadratic-assignment arithmetic",
        ))
    }
}
/// Termination of a cooperative execution stop.
fn execution_stop(reason: ExecutionStopReason) -> OptimizeTermination {
    match reason {
        ExecutionStopReason::DeadlineReached => OptimizeTermination::TimeLimit,
        _ => OptimizeTermination::Cancelled,
    }
}
/// Termination of a nested work stop, absent for any other error.
fn work_stop(error: &SciOxideError) -> Option<OptimizeTermination> {
    match error.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        } => Some(OptimizeTermination::IterationLimit),
        scioxide_core::SciOxideErrorView::ExecutionStopped { reason } => {
            Some(execution_stop(reason))
        }
        _ => None,
    }
}
/// Why one restart or the whole heuristic stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Completed,
    Method,
    Budget(OptimizeTermination),
}
impl Stop {
    /// Completed stages converge; method limits are iteration limits; budget
    /// stops report the actual exhausted resource or execution stop.
    const fn termination(self) -> OptimizeTermination {
        match self {
            Self::Completed => OptimizeTermination::Converged,
            Self::Method => OptimizeTermination::IterationLimit,
            Self::Budget(termination) => termination,
        }
    }
}
use crate::validation::number;
fn score<S: RealScalar + NumericScalar<Real = S>>(
    a: &[S],
    b: &[S],
    n: usize,
    p: &[usize],
) -> Result<S, SciOxideError> {
    let mut sum = RealProductSum::<S>::new();
    for i in 0..n {
        for j in 0..n {
            sum.add_product(a[i * n + j], b[p[i] * n + p[j]])?;
        }
    }
    sum.finish()
}

/// Minimize or maximize a quadratic on the closed unit interval.
fn line_step<S: RealScalar>(
    f0: S,
    f1: S,
    derivative: S,
    curvature: S,
    direction: QapDirection,
) -> Result<S, SciOxideError> {
    let signed_curvature = if direction == QapDirection::Minimize {
        curvature
    } else {
        -curvature
    };
    if signed_curvature > S::ZERO {
        Ok(finite(-derivative / S::TWO / curvature)?
            .max(S::ZERO)
            .min(S::ONE))
    } else if better(f1, f0, direction) {
        Ok(S::ONE)
    } else {
        Ok(S::ZERO)
    }
}

/// Selects whether [`QapPlan`] minimizes or maximizes the objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QapDirection {
    /// Minimize the objective, for example a cost.
    Minimize,
    /// Maximize the objective, for example the similarity of two graphs.
    Maximize,
}
/// Selects the heuristic of [`QapPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QapMethod {
    /// Fast approximate QAP: Frank–Wolfe iterations on the relaxation over
    /// doubly stochastic matrices, then projection onto a permutation.
    Faq,
    /// Local search that swaps pairs of assignments while the objective improves.
    TwoOpt,
    /// FAQ, then 2-opt refinement of its permutation.
    FaqThenTwoOpt,
}

/// Borrows the matrices of a quadratic assignment problem.
///
/// For `n × n` matrices `A` and `B`, the problem is to find a permutation `p` of
/// `0..n` that minimizes or maximizes
///
/// ```text
/// f(p) = Σᵢ Σⱼ A[i, j] · B[p(i), p(j)]
/// ```
///
/// so `p(i)` is the node of `B` assigned to node `i` of `A`. A fixed pair
/// `(i, j)` requires `p(i) = j`. [`Self::prepare`] turns the view into a
/// [`QapPlan`].
#[derive(Clone, Copy, Debug)]
pub struct QapProblemView<'a, S: RealScalar + NumericScalar<Real = S>> {
    a: MatrixView<'a, S>,
    b: MatrixView<'a, S>,
    fixed: &'a [(usize, usize)],
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> QapProblemView<'a, S> {
    /// Creates a view of `a`, `b` and the fixed pairs `fixed`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `a` and `b` are not square matrices of the same
    ///   order, or a fixed pair is out of range or repeats a node.
    /// - `NonFinite` if a matrix entry is NaN or infinite.
    pub fn new(
        a: MatrixView<'a, S>,
        b: MatrixView<'a, S>,
        fixed: &'a [(usize, usize)],
    ) -> Result<Self, SciOxideError> {
        if a.rows() != a.cols() || b.rows() != a.rows() || b.cols() != a.cols() {
            return Err(invalid(
                scioxide_core::argument!("b"),
                "equal square matrices required",
            ));
        }
        let n = a.rows();
        for i in 0..n {
            for j in 0..n {
                if !a.get(i, j)?.is_finite() || !b.get(i, j)?.is_finite() {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "matrix entries must be finite",
                    ));
                }
            }
        }
        validate_fixed(n, fixed)?;
        Ok(Self { a, b, fixed })
    }
    /// Copies the problem into a [`QapPlan`].
    ///
    /// To limit the storage before anything is allocated, use
    /// [`Self::prepare_with_limit`].
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn prepare(self) -> Result<QapPlan<S>, SciOxideError> {
        self.prepare_with_limit(usize::MAX)
    }
    /// Copies the problem into a [`QapPlan`] if the plan and its workspace fit in `max_bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` if they need more than `max_bytes` bytes, and
    /// the errors of [`Self::prepare`].
    pub fn prepare_with_limit(self, max_bytes: usize) -> Result<QapPlan<S>, SciOxideError> {
        let n = self.a.rows();
        if required_bytes::<S>(n, self.fixed.len())? > max_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("qap_workspace_bytes"),
                required_bytes::<S>(n, self.fixed.len())?,
                max_bytes,
            ));
        }
        let len = checked_mul(n, n, scioxide_core::quantity!("matrix_entries"))?;
        let mut a = try_filled(len, S::ZERO, scioxide_core::quantity!("plan.a"))?;
        let mut b = try_filled(len, S::ZERO, scioxide_core::quantity!("plan.b"))?;
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = self.a.get(i, j)?;
                b[i * n + j] = self.b.get(i, j)?;
            }
        }
        QapPlan::from_dense_parts(n, a, b, self.fixed)
    }
}
fn validate_fixed(n: usize, fixed: &[(usize, usize)]) -> Result<(), SciOxideError> {
    if fixed.len() > n {
        return Err(invalid(
            scioxide_core::argument!("fixed"),
            "too many fixed pairs",
        ));
    }
    for (i, &(row, col)) in fixed.iter().enumerate() {
        if row >= n || col >= n || fixed[..i].iter().any(|&(r, c)| r == row || c == col) {
            return Err(invalid(
                scioxide_core::argument!("fixed"),
                "fixed pairs must be injective and in range",
            ));
        }
    }
    Ok(())
}
fn required_bytes<S: RealScalar>(n: usize, fixed: usize) -> Result<usize, SciOxideError> {
    let k = n
        .checked_sub(fixed)
        .ok_or_else(|| invalid(scioxide_core::argument!("fixed"), "too many fixed pairs"))?;
    let n2 = checked_mul(n, n, scioxide_core::quantity!("matrix_entries"))?;
    let k2 = checked_mul(k, k, scioxide_core::quantity!("free_matrix_entries"))?;
    let scalar = core::mem::size_of::<S>();
    let index = core::mem::size_of::<usize>();
    let matrices = checked_mul(
        checked_mul(6, n2, scioxide_core::quantity!("matrix_bytes"))?,
        scalar,
        scioxide_core::quantity!("matrix_bytes"),
    )?;
    let scratch = checked_mul(
        checked_mul(16, k2, scioxide_core::quantity!("scratch_bytes"))?,
        checked_add(scalar, index, scioxide_core::quantity!("scratch_bytes"))?,
        scioxide_core::quantity!("scratch_bytes"),
    )?;
    let vectors = checked_mul(
        checked_mul(
            64,
            checked_add(n, 1, scioxide_core::quantity!("vector_bytes"))?,
            scioxide_core::quantity!("vector_bytes"),
        )?,
        checked_add(
            checked_add(scalar, index, scioxide_core::quantity!("vector_bytes"))?,
            1,
            scioxide_core::quantity!("vector_bytes"),
        )?,
        scioxide_core::quantity!("vector_bytes"),
    )?;
    checked_add(
        checked_add(
            matrices,
            scratch,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        checked_add(
            vectors,
            faq_requirements::<S>(k)?.required_bytes::<S>()?,
            scioxide_core::quantity!("workspace_bytes"),
        )?,
        scioxide_core::quantity!("workspace_bytes"),
    )
}

/// Finds good permutations for a quadratic assignment problem by heuristics.
///
/// A plan holds the matrices and fixed pairs of a [`QapProblemView`], or of
/// sparse matrices ([`Self::from_sparse`]); the objective is defined on
/// [`QapProblemView`]. Preparation copies both matrices and constructs the
/// free-node blocks reused by FAQ gradients in `O(n²)` work and storage.
/// [`QapMethod`] selects the heuristic:
///
/// - FAQ (fast approximate QAP) runs Frank–Wolfe iterations on the relaxation
///   over doubly stochastic matrices `P`. Each iteration forms the gradient
///   `A P Bᵀ + Aᵀ P B` with two matrix products, O(n³) work, solves a linear
///   assignment problem for the search direction, and minimizes the quadratic
///   objective exactly along it. A final linear assignment projects the result
///   onto a permutation.
/// - 2-opt swaps pairs of assignments while the objective improves.
///
/// Further restarts ([`QapOptions::with_restarts`]) begin from seeded random
/// permutations, and the best result is kept. The returned permutation is
/// checked and its objective recomputed from the original matrices. Both
/// methods are heuristics, so no termination status certifies a global optimum.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_optimize::{QapDirection, QapProblemView};
/// let a = [0.0_f64, 3.0, 1.0, 8.0, 0.0, 2.0, 4.0, 6.0, 0.0];
/// let b = [0.0, 5.0, 7.0, 2.0, 0.0, 9.0, 3.0, 1.0, 0.0];
/// let (a, b) = (MatrixView::from_slice(3, 3, &a)?, MatrixView::from_slice(3, 3, &b)?);
/// let plan = QapProblemView::new(a, b, &[(1, 2)])?.prepare()?;
/// let result = plan.solve(QapDirection::Minimize, None)?;
/// assert_eq!(result.permutation(), Some(&[1, 2, 0][..]));
/// assert!((result.objective().unwrap() - 105.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. T. Vogelstein et al., "Fast approximate quadratic programming for graph
///   matching", *PLoS ONE* 10(4), e0121002, 2015.
/// - E. M. Loiola et al., "A survey for the quadratic assignment problem",
///   *Eur. J. Oper. Res.* 176(2), 657–690, 2007.
#[derive(Debug, Clone)]
pub struct QapPlan<S: RealScalar + NumericScalar<Real = S>> {
    n: usize,
    a: Vec<S>,
    b: Vec<S>,
    fixed: Vec<(usize, usize)>,
    free_a: Vec<usize>,
    free_b: Vec<usize>,
    // Vertical [A_free; A_free^T] and [B_free^T; B_free] blocks.
    gradient_left: Vec<S>,
    gradient_right: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> QapPlan<S> {
    fn from_dense_parts(
        n: usize,
        a: Vec<S>,
        b: Vec<S>,
        fixed: &[(usize, usize)],
    ) -> Result<Self, SciOxideError> {
        validate_fixed(n, fixed)?;
        let mut free_a = try_filled(0, 0, scioxide_core::quantity!("free_a"))?;
        let mut free_b = try_filled(0, 0, scioxide_core::quantity!("free_b"))?;
        free_a.try_reserve(n).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("free_a"), n)
        })?;
        free_b.try_reserve(n).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("free_b"), n)
        })?;
        for i in 0..n {
            if !fixed.iter().any(|&(r, _)| r == i) {
                free_a.push(i);
            }
            if !fixed.iter().any(|&(_, c)| c == i) {
                free_b.push(i);
            }
        }
        let k = free_a.len();
        let k2 = checked_mul(k, k, scioxide_core::quantity!("free_matrix_entries"))?;
        let block_len = checked_mul(2, k2, scioxide_core::quantity!("faq_block_entries"))?;
        let mut gradient_left = try_filled(
            block_len,
            S::ZERO,
            scioxide_core::quantity!("plan.gradient_left"),
        )?;
        let mut gradient_right = try_filled(
            block_len,
            S::ZERO,
            scioxide_core::quantity!("plan.gradient_right"),
        )?;
        for i in 0..k {
            for j in 0..k {
                gradient_left[i * k + j] = a[free_a[i] * n + free_a[j]];
                gradient_left[k2 + i * k + j] = a[free_a[j] * n + free_a[i]];
                gradient_right[i * k + j] = b[free_b[j] * n + free_b[i]];
                gradient_right[k2 + i * k + j] = b[free_b[i] * n + free_b[j]];
            }
        }
        Ok(Self {
            n,
            a,
            b,
            fixed: try_copy_slice(fixed, scioxide_core::quantity!("fixed"))?,
            free_a,
            free_b,
            gradient_left,
            gradient_right,
        })
    }
    /// Creates a plan from sparse matrices; absent entries are zero.
    ///
    /// To limit the storage before anything is allocated, use
    /// [`Self::from_sparse_with_limit`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the matrices are not square of the same order, or a
    ///   fixed pair is out of range or repeats a node.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn from_sparse(
        a: SparseView<'_, S>,
        b: SparseView<'_, S>,
        fixed: &[(usize, usize)],
    ) -> Result<Self, SciOxideError> {
        Self::from_sparse_with_limit(a, b, fixed, usize::MAX)
    }
    /// Creates a plan from sparse matrices if the plan and its workspace fit in `max_bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` if they need more than `max_bytes` bytes, and
    /// the errors of [`Self::from_sparse`].
    pub fn from_sparse_with_limit(
        a: SparseView<'_, S>,
        b: SparseView<'_, S>,
        fixed: &[(usize, usize)],
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        if a.rows() != a.cols() || b.rows() != a.rows() || b.cols() != a.cols() {
            return Err(invalid(
                scioxide_core::argument!("b"),
                "equal square matrices required",
            ));
        }
        let n = a.rows();
        validate_fixed(n, fixed)?;
        if required_bytes::<S>(n, fixed.len())? > max_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("qap_workspace_bytes"),
                required_bytes::<S>(n, fixed.len())?,
                max_bytes,
            ));
        }
        let len = checked_mul(n, n, scioxide_core::quantity!("matrix_entries"))?;
        let mut left = try_filled(len, S::ZERO, scioxide_core::quantity!("plan.a"))?;
        let mut right = try_filled(len, S::ZERO, scioxide_core::quantity!("plan.b"))?;
        for (i, j, v) in a.iter() {
            if !v.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "matrix entries must be finite",
                ));
            }
            left[i * n + j] = v;
        }
        for (i, j, v) in b.iter() {
            if !v.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "matrix entries must be finite",
                ));
            }
            right[i * n + j] = v;
        }
        Self::from_dense_parts(n, left, right, fixed)
    }
    /// Returns the order `n` of the problem.
    pub const fn len(&self) -> usize {
        self.n
    }
    /// Returns whether the order is zero.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// Returns the objective of `permutation`, computed from the original matrices.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `permutation` has the wrong length or violates a
    /// fixed pair, or an error if the sum overflows.
    pub fn objective(&self, permutation: PermutationView<'_>) -> Result<S, SciOxideError> {
        let p = permutation.new_to_old();
        self.check_start(p, scioxide_core::argument!("permutation"))?;
        score(&self.a, &self.b, self.n, p)
    }
    /// Check a start permutation supplied as the entry point's `argument`.
    fn check_start(&self, p: &[usize], argument: Argument) -> Result<(), SciOxideError> {
        if p.len() != self.n {
            return Err(invalid(argument, "initial permutation length"));
        }
        if self.fixed.iter().any(|&(i, j)| p[i] != j) {
            return Err(invalid(
                argument,
                "initial permutation conflicts with fixed pairs",
            ));
        }
        Ok(())
    }
    /// Returns an upper bound on the bytes of the plan and one workspace.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the computation overflows.
    pub fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        required_bytes::<S>(self.n, self.fixed.len())
    }
    /// Reject a zero or exceeded workspace payload cap.
    fn check_capacity(&self, options: &QapOptions<S>, copies: usize) -> Result<(), SciOxideError> {
        if options.max_workspace_bytes == 0 {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "workspace byte capacity must be positive",
            ));
        }
        let required = checked_mul(
            self.workspace_bytes()?,
            copies,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if required > options.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("qap_workspace_bytes"),
                required,
                options.max_workspace_bytes,
            ));
        }
        Ok(())
    }
    /// Creates a workspace for solves with `options`.
    ///
    /// The workspace holds all numerical storage, so sequential solves with it
    /// do not allocate, including the first one and all restarts.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the storage cap of `options` is zero.
    /// - `ResourceLimit` if the plan and workspace need more than the cap.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, options: &QapOptions<S>) -> Result<QapWorkspace<S>, SciOxideError> {
        self.check_capacity(options, 1)?;
        let k = self.free_a.len();
        let k2 = checked_mul(k, k, scioxide_core::quantity!("free_matrix_entries"))?;
        let block_len = checked_mul(2, k2, scioxide_core::quantity!("faq_block_entries"))?;
        let mut kernels = LinalgWorkspace::new();
        kernels.try_reserve(&faq_requirements::<S>(k)?)?;
        let assignment = AssignmentPlan::from_dense(k)?;
        let assignment_work = assignment.workspace(())?;
        Ok(QapWorkspace {
            p: try_filled(k2, S::ZERO, scioxide_core::quantity!("workspace.p"))?,
            q: try_filled(k2, S::ZERO, scioxide_core::quantity!("workspace.q"))?,
            gradient: try_filled(k2, S::ZERO, scioxide_core::quantity!("workspace.gradient"))?,
            products: try_filled(
                block_len,
                S::ZERO,
                scioxide_core::quantity!("workspace.products"),
            )?,
            joined_products: try_filled(
                block_len,
                S::ZERO,
                scioxide_core::quantity!("workspace.joined_products"),
            )?,
            kernels,
            linear: try_filled(k2, S::ZERO, scioxide_core::quantity!("workspace.linear"))?,
            perm: try_filled(self.n, 0, scioxide_core::quantity!("workspace.perm"))?,
            candidate: try_filled(self.n, 0, scioxide_core::quantity!("workspace.candidate"))?,
            best: try_filled(self.n, 0, scioxide_core::quantity!("workspace.best"))?,
            assignment,
            assignment_work,
            shell: crate::attempt::Shell::new(),
        })
    }
    /// Internal identity binding this plan and immutable options to an epoch.
    fn identity(
        &self,
        direction: QapDirection,
        options: &QapOptions<S>,
        token: ProblemToken,
    ) -> SolveIdentity {
        SolveIdentity::new(
            token,
            scioxide_core::algorithm_id!("optimize", "quadratic_assignment", 1),
            self.n as u128,
            crate::context::options_token(&(direction, options)),
        )
    }
    /// Solves with FAQ followed by 2-opt and the default [`QapOptions`].
    ///
    /// Returns a partial result if an iteration or swap limit is reached.
    /// Preparation stores both matrices and free-node blocks in `O(n²)` space.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_with_options`].
    pub fn solve(
        &self,
        direction: QapDirection,
        initial: Option<PermutationView<'_>>,
    ) -> Result<QapResult<S>, SciOxideError>
    where
        S: Send + Sync,
    {
        self.solve_with_options(direction, initial, &QapOptions::default())
    }

    /// Runs the heuristic on the calling thread and returns a new result.
    ///
    /// `initial` is an optional starting permutation for the first restart.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace`] and [`Self::solve_into`].
    pub fn solve_with_options(
        &self,
        direction: QapDirection,
        initial: Option<PermutationView<'_>>,
        options: &QapOptions<S>,
    ) -> Result<QapResult<S>, SciOxideError>
    where
        S: Send + Sync,
    {
        let mut workspace = self.workspace(options)?;
        let mut result = QapResult::empty(self.n)?;
        self.solve_into(
            direction,
            initial,
            options,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(result)
    }
    /// Runs the heuristic into `result`, reusing `workspace`.
    ///
    /// `initial` is an optional starting permutation for the first restart.
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent solve context. FAQ iterations and swap checks are counted as work
    /// of this solve. The limits in `options`, cancellation and deadlines stop
    /// the heuristic. With [`ExecutionPolicy::Sequential`] (the default),
    /// restarts run in order in `workspace` and do not allocate. Any other policy
    /// runs the restarts through the executor of `scope`, with one workspace and
    /// result per restart that must all fit in the storage cap, and combines them
    /// in restart order.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the storage cap is zero, `result` or `workspace` has
    ///   the wrong size, or `initial` has the wrong length or violates a fixed
    ///   pair.
    /// - `ResourceLimit` if the plan needs more storage than the cap.
    /// - An error if a numerical failure, cancellation or deadline occurs before
    ///   any restart completes.
    pub fn solve_into<'scope, 'capability>(
        &self,
        direction: QapDirection,
        initial: Option<PermutationView<'_>>,
        options: &QapOptions<S>,
        result: &mut QapResult<S>,
        workspace: &mut QapWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        S: Send + Sync,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let start = initial.map(PermutationView::new_to_old);
        if let Some(p) = start {
            self.check_start(p, scioxide_core::argument!("initial"))?;
        }
        let sequential = options.execution == ExecutionPolicy::Sequential;
        let copies = if sequential {
            1
        } else {
            checked_add(
                options.restarts,
                1,
                scioxide_core::quantity!("restart_count"),
            )?
        };
        self.check_capacity(options, copies)?;
        if result.permutation.len() != self.n
            || workspace.perm.len() != self.n
            || workspace.p.len()
                != checked_mul(
                    self.free_a.len(),
                    self.free_a.len(),
                    scioxide_core::quantity!("free_matrix_entries"),
                )?
        {
            return Err(invalid(
                scioxide_core::argument!("result"),
                "result or workspace shape",
            ));
        }
        crate::attempt::preflight(&scope, None)?;
        let mut shell = core::mem::take(&mut workspace.shell);
        let (outcome, started) = shell.execute(
            scope,
            crate::attempt::Entry::Call {
                identity: self.identity(direction, options, ProblemToken::new(0)),
                budgets: SolveBudgets::unbounded(),
            },
            |context| {
                if sequential {
                    self.sequential(direction, options, start, result, workspace, context)
                } else {
                    self.partitioned(direction, options, start, result, context)
                }
            },
        );
        let succeeded = outcome.is_ok() && result.termination == OptimizeTermination::Converged;
        shell.finish(
            AttemptKind::Call,
            started,
            outcome.is_err(),
            succeeded,
            outcome.is_ok().then_some(()),
        );
        workspace.shell = shell;
        outcome
    }

    /// Run every restart in order in the prepared workspace.
    fn sequential(
        &self,
        direction: QapDirection,
        options: &QapOptions<S>,
        start: Option<&[usize]>,
        result: &mut QapResult<S>,
        w: &mut QapWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let k = self.free_a.len();
        let mut best_value = None;
        let mut selected = 0;
        let mut termination = Stop::Completed;
        let mut fw_total = 0;
        let mut swap_total = 0;
        let mut last_relaxed = None;
        let mut last_gap = None;
        let fixed_value = if options.method != QapMethod::TwoOpt && k > 0 {
            self.fixed_terms(&mut w.linear)?
        } else {
            S::ZERO
        };
        let mut kernels = LinalgExecutionContext::from(context.execution());
        'restarts: for restart in 0..=options.restarts {
            if let Some(reason) = context.execution().stop_reason()? {
                termination = Stop::Budget(execution_stop(reason));
                break;
            }
            self.initialize_permutation(
                restart,
                options.seed,
                start,
                &mut w.perm,
                &mut w.candidate,
            )?;
            let mut relaxed = None;
            let mut gap = None;
            if options.method != QapMethod::TwoOpt && k > 0 {
                self.initialize_relaxation(restart, start, &w.perm, &mut w.p)?;
                let mut converged = false;
                for _ in 0..options.max_fw_iterations.get() {
                    if !context.record_iteration()?.is_granted() {
                        termination = Stop::Budget(OptimizeTermination::IterationLimit);
                        break;
                    }
                    fw_total = checked_add(fw_total, 1, scioxide_core::quantity!("fw_iterations"))?;
                    let f0 = match self.gradient(w, fixed_value, kernels.reborrow()) {
                        Ok(value) => value,
                        Err(error) => match work_stop(&error) {
                            Some(stop) if best_value.is_some() => {
                                termination = Stop::Budget(stop);
                                break 'restarts;
                            }
                            _ => return Err(error),
                        },
                    };
                    let assignment_direction = if direction == QapDirection::Minimize {
                        AssignmentObjective::Minimize
                    } else {
                        AssignmentObjective::Maximize
                    };
                    let solved = match w.assignment.solve_scoped(
                        &w.gradient,
                        assignment_direction,
                        GraphExecutionOptions::new(),
                        &mut w.assignment_work,
                        &mut *context,
                    ) {
                        Ok(value) => value,
                        Err(error) => match work_stop(&error) {
                            Some(stop) if best_value.is_some() => {
                                termination = Stop::Budget(stop);
                                break 'restarts;
                            }
                            _ => return Err(error),
                        },
                    };
                    let AssignmentOutcome::Full(assignment) = solved else {
                        return Err(invalid(
                            scioxide_core::argument!("self"),
                            "complete assignment unexpectedly infeasible",
                        ));
                    };
                    w.q.fill(S::ZERO);
                    for (i, &column) in assignment.left.iter().enumerate() {
                        let Some(j) = column else {
                            return Err(invalid(
                                scioxide_core::argument!("self"),
                                "assignment omitted a row",
                            ));
                        };
                        w.q[i * k + j] = S::ONE;
                    }
                    let mut derivative = RealProductSum::<S>::new();
                    for ((&g, &q), &p) in w.gradient.iter().zip(&w.q).zip(&w.p) {
                        derivative.add_product(g, finite(q - p)?)?;
                    }
                    let deriv = derivative.finish()?;
                    let f1 = self.assignment_objective(assignment.left, &w.linear, fixed_value)?;
                    let a = finite(finite(f1 - f0)? - deriv)?;
                    let signed_deriv = if direction == QapDirection::Minimize {
                        deriv
                    } else {
                        -deriv
                    };
                    gap = Some(if signed_deriv < S::ZERO {
                        -signed_deriv
                    } else {
                        S::ZERO
                    });
                    let t = line_step(f0, f1, deriv, a, direction)?;
                    let mut norm = 0.0_f64;
                    for index in 0..k * k {
                        let delta = t.into_f64() * (w.q[index].into_f64() - w.p[index].into_f64());
                        norm += delta * delta;
                        w.p[index] =
                            finite(w.p[index] + finite(t * finite(w.q[index] - w.p[index])?)?)?;
                    }
                    relaxed = Some(line_value(f0, f1, deriv, a, t)?);
                    if norm.sqrt() / (k as f64).sqrt() <= options.step_threshold()?.into_f64() {
                        converged = true;
                        break;
                    }
                }
                if matches!(termination, Stop::Budget(_)) {
                    break;
                }
                if !converged {
                    termination = Stop::Method;
                }
                let projected = match w.assignment.solve_scoped(
                    &w.p,
                    AssignmentObjective::Maximize,
                    GraphExecutionOptions::new(),
                    &mut w.assignment_work,
                    &mut *context,
                ) {
                    Ok(value) => value,
                    Err(error) => match work_stop(&error) {
                        Some(stop) if best_value.is_some() => {
                            termination = Stop::Budget(stop);
                            break 'restarts;
                        }
                        _ => return Err(error),
                    },
                };
                let AssignmentOutcome::Full(projected) = projected else {
                    return Err(invalid(
                        scioxide_core::argument!("self"),
                        "complete projection unexpectedly infeasible",
                    ));
                };
                for (i, &row) in self.free_a.iter().enumerate() {
                    let Some(column) = projected.left[i] else {
                        return Err(invalid(
                            scioxide_core::argument!("self"),
                            "projection omitted a row",
                        ));
                    };
                    w.perm[row] = self.free_b[column];
                }
            }
            let mut value = score(&self.a, &self.b, self.n, &w.perm)?;
            let mut stop_after_restart = false;
            if options.method != QapMethod::Faq && k > 1 {
                let (new_value, checks, stop) =
                    self.refine(direction, &mut w.perm, value, options, context)?;
                value = new_value;
                swap_total =
                    checked_add(swap_total, checks, scioxide_core::quantity!("swap_checks"))?;
                if stop != Stop::Completed {
                    termination = stop;
                    stop_after_restart = matches!(stop, Stop::Budget(_));
                }
            }
            if best_value.is_none_or(|old| {
                better(value, old, direction)
                    || (value == old && w.perm.as_slice() < w.best.as_slice())
            }) {
                w.best.copy_from_slice(&w.perm);
                best_value = Some(value);
                selected = restart;
                last_relaxed = relaxed;
                last_gap = gap;
            }
            if stop_after_restart {
                break;
            }
        }
        let Some(objective) = best_value else {
            return Err(invalid(
                scioxide_core::argument!("context"),
                "no completed quadratic-assignment candidate",
            ));
        };
        result.permutation.copy_from_slice(&w.best);
        result.objective = Some(objective);
        result.relaxation_objective = last_relaxed;
        result.fw_gap = last_gap;
        result.fw_iterations = fw_total;
        result.swap_checks = swap_total;
        result.restart = selected;
        result.termination = termination.termination();
        Ok(())
    }
    /// Execute independent restarts through the context's executor, reducing in restart order.
    fn partitioned(
        &self,
        direction: QapDirection,
        options: &QapOptions<S>,
        supplied: Option<&[usize]>,
        result: &mut QapResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        S: Send + Sync,
    {
        let policy = options.execution;
        let count = checked_add(
            options.restarts,
            1,
            scioxide_core::quantity!("restart_count"),
        )?;
        let mut child_options = options.clone();
        child_options.restarts = 0;
        child_options.execution = ExecutionPolicy::Sequential;
        let mut children = try_with_capacity(count, scioxide_core::quantity!("children"))?;
        let mut workspaces = try_with_capacity(count, scioxide_core::quantity!("workspaces"))?;
        let mut reports = try_with_capacity(count, scioxide_core::quantity!("reports"))?;
        let mut starts = try_with_capacity(count, scioxide_core::quantity!("starts"))?;
        let mut errors = try_with_capacity(count, scioxide_core::quantity!("errors"))?;
        for index in 0..count {
            children.push(PartitionedSolve::default());
            workspaces.push(self.workspace(&child_options)?);
            reports.push(QapResult::empty(self.n)?);
            errors.push(None);
            if index == 0 {
                starts.push(
                    supplied
                        .map(|p| try_copy_slice(p, scioxide_core::quantity!("initial")))
                        .transpose()?,
                );
            } else {
                let mut permutation =
                    try_filled(self.n, 0, scioxide_core::quantity!("permutation"))?;
                let mut scratch =
                    try_filled(self.n, 0, scioxide_core::quantity!("permutation_scratch"))?;
                self.initialize_permutation(
                    index,
                    options.seed,
                    None,
                    &mut permutation,
                    &mut scratch,
                )?;
                starts.push(Some(permutation));
            }
        }
        with_partitioned_solves(
            context,
            &mut children,
            SolveBudgets::unbounded(),
            |capabilities, children| {
                let plan = capabilities.plan(
                    policy,
                    count,
                    scioxide_core::PlanLimits {
                        work_per_item: core::num::NonZeroUsize::MIN,
                        auto_threshold: core::num::NonZeroUsize::MIN,
                        worker_limit: core::num::NonZeroUsize::MIN.saturating_add(63),
                    },
                );
                let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
                let mut remaining_reports = reports.as_mut_slice();
                let mut remaining_work = workspaces.as_mut_slice();
                let mut remaining_children = children;
                let mut remaining_starts = starts.as_mut_slice();
                let mut remaining_errors = errors.as_mut_slice();
                for partition in plan.partitions() {
                    let length = partition.range().len();
                    let (r, tail) = remaining_reports.split_at_mut(length);
                    remaining_reports = tail;
                    let (w, tail) = remaining_work.split_at_mut(length);
                    remaining_work = tail;
                    let (c, tail) = remaining_children.split_at(length)?;
                    remaining_children = tail;
                    let (s, tail) = remaining_starts.split_at_mut(length);
                    remaining_starts = tail;
                    let (e, tail) = remaining_errors.split_at_mut(length);
                    remaining_errors = tail;
                    slots[partition.worker_index()] = PartitionSlot::new((r, w, c, s, e));
                }
                capabilities.execute_partitioned(
                    plan,
                    &slots[..plan.worker_count()],
                    &|partition, (r, w, c, s, e), _| {
                        for (index, ((((report, work), child), start), error)) in
                            partition.range().zip(
                                r.iter_mut()
                                    .zip(w.iter_mut())
                                    .zip(c)
                                    .zip(s.iter())
                                    .zip(e.iter_mut()),
                            )
                        {
                            let identity = self.identity(
                                direction,
                                &child_options,
                                ProblemToken::new(index as u128),
                            );
                            *error = child
                                .run(capabilities, identity, |ctx| {
                                    self.sequential(
                                        direction,
                                        &child_options,
                                        start.as_deref(),
                                        report,
                                        work,
                                        ctx,
                                    )
                                })
                                .err();
                        }
                        Ok(())
                    },
                )?;

                Ok(())
            },
        )?;
        if let Some(error) = errors.into_iter().flatten().next() {
            return Err(error);
        }
        let mut selected: Option<usize> = None;
        let mut fw_total = 0_usize;
        let mut swap_total = 0_usize;
        let mut termination = OptimizeTermination::Converged;
        for (index, report) in reports.iter().enumerate() {
            fw_total = checked_add(
                fw_total,
                report.fw_iterations,
                scioxide_core::quantity!("fw_iterations"),
            )?;
            swap_total = checked_add(
                swap_total,
                report.swap_checks,
                scioxide_core::quantity!("swap_checks"),
            )?;
            if report.termination != OptimizeTermination::Converged {
                termination = report.termination;
            }
            let value = report.objective.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("context"),
                    "child omitted QAP result",
                )
            })?;
            let choose = if let Some(best) = selected {
                let current = reports[best].objective.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("context"),
                        "child omitted QAP result",
                    )
                })?;
                better(value, current, direction)
                    || (value == current && report.permutation < reports[best].permutation)
            } else {
                true
            };
            if choose {
                selected = Some(index);
            }
        }
        let index = selected.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("context"),
                "no completed quadratic-assignment candidate",
            )
        })?;
        let best = &reports[index];
        result.permutation.copy_from_slice(&best.permutation);
        result.objective = best.objective;
        result.relaxation_objective = best.relaxation_objective;
        result.fw_gap = best.fw_gap;
        result.fw_iterations = fw_total;
        result.swap_checks = swap_total;
        result.restart = index;
        result.termination = termination;
        Ok(())
    }
    fn initialize_permutation(
        &self,
        restart: usize,
        seed: u64,
        start: Option<&[usize]>,
        p: &mut [usize],
        scratch: &mut [usize],
    ) -> Result<(), SciOxideError> {
        for &(i, j) in &self.fixed {
            p[i] = j;
        }
        if restart == 0
            && let Some(start) = start
        {
            p.copy_from_slice(start);
            return Ok(());
        }
        let cols = &mut scratch[..self.free_b.len()];
        cols.copy_from_slice(&self.free_b);
        if restart > 0 {
            let mut rng = Xoshiro256StarStar::from_seed(seed.wrapping_add(restart as u64));
            for i in (1..cols.len()).rev() {
                let j = uniform_index(i + 1, &mut rng)?;
                cols.swap(i, j);
            }
        }
        for (&row, &column) in self.free_a.iter().zip(cols.iter()) {
            p[row] = column;
        }
        Ok(())
    }
    fn initialize_relaxation(
        &self,
        restart: usize,
        start: Option<&[usize]>,
        permutation: &[usize],
        p: &mut [S],
    ) -> Result<(), SciOxideError> {
        let k = self.free_a.len();
        if k == 0 {
            return Ok(());
        }
        let uniform = number::<S>(1.0 / k as f64)?;
        p.fill(uniform);
        if restart > 0 || start.is_some() {
            let half = number::<S>(0.5)?;
            for value in p.iter_mut() {
                *value = finite(*value * half)?;
            }
            for (i, &row) in self.free_a.iter().enumerate() {
                let j = self
                    .free_b
                    .iter()
                    .position(|&c| c == permutation[row])
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("initial"),
                            "start does not map free nodes",
                        )
                    })?;
                p[i * k + j] = finite(p[i * k + j] + half)?;
            }
        }
        Ok(())
    }
    /// Seed interactions are independent of the relaxation and all restarts.
    fn fixed_terms(&self, linear: &mut [S]) -> Result<S, SciOxideError> {
        if self.fixed.is_empty() {
            linear.fill(S::ZERO);
            return Ok(S::ZERO);
        }
        let k = self.free_a.len();
        let n = self.n;
        for (i, &row) in self.free_a.iter().enumerate() {
            for (j, &col) in self.free_b.iter().enumerate() {
                let mut term = RealProductSum::<S>::new();
                for &(r, c) in &self.fixed {
                    term.add_product(self.a[r * n + row], self.b[c * n + col])?;
                    term.add_product(self.a[row * n + r], self.b[col * n + c])?;
                }
                linear[i * k + j] = term.finish()?;
            }
        }
        let mut constant = RealProductSum::<S>::new();
        for &(i, pi) in &self.fixed {
            for &(j, pj) in &self.fixed {
                constant.add_product(self.a[i * n + j], self.b[pi * n + pj])?;
            }
        }
        constant.finish()
    }

    /// Two rectangular GEMMs form the general asymmetric gradient; reductions
    /// reuse the quadratic part before adding the seeded linear term.
    fn gradient(
        &self,
        w: &mut QapWorkspace<S>,
        fixed_value: S,
        mut context: LinalgExecutionContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        let k = self.free_a.len();
        let k2 = k * k;
        MatrixView::from_slice(2 * k, k, &self.gradient_left)?.mul_mat_into(
            MatrixView::from_slice(k, k, &w.p)?,
            &FAQ_KERNELS,
            MatrixViewMut::from_slice(2 * k, k, &mut w.products)?,
            &mut w.kernels,
            context.reborrow(),
        )?;
        for i in 0..k {
            let row = &mut w.joined_products[i * 2 * k..(i + 1) * 2 * k];
            row[..k].copy_from_slice(&w.products[i * k..(i + 1) * k]);
            row[k..].copy_from_slice(&w.products[k2 + i * k..k2 + (i + 1) * k]);
        }
        MatrixView::from_slice(k, 2 * k, &w.joined_products)?.mul_mat_into(
            MatrixView::from_slice(2 * k, k, &self.gradient_right)?,
            &FAQ_KERNELS,
            MatrixViewMut::from_slice(k, k, &mut w.gradient)?,
            &mut w.kernels,
            context,
        )?;
        let mut objective = RealProductSum::<S>::new();
        objective.add_product(fixed_value, S::ONE)?;
        for ((g, &p), &linear) in w.gradient.iter_mut().zip(&w.p).zip(&w.linear) {
            objective.add_product(*g / S::TWO, p)?;
            objective.add_product(linear, p)?;
            *g = finite(*g + linear)?;
        }
        objective.finish()
    }

    /// A permutation endpoint requires O(k^2) work, never another GEMM.
    fn assignment_objective(
        &self,
        columns: &[Option<usize>],
        linear: &[S],
        fixed_value: S,
    ) -> Result<S, SciOxideError> {
        let k = self.free_a.len();
        let b = &self.gradient_right[k * k..];
        let mut objective = RealProductSum::<S>::new();
        objective.add_product(fixed_value, S::ONE)?;
        for (i, &column) in columns.iter().enumerate() {
            let column = column.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("assignment"),
                    "assignment omitted a row",
                )
            })?;
            objective.add_product(linear[i * k + column], S::ONE)?;
            for (j, &other) in columns.iter().enumerate() {
                let other = other.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("assignment"),
                        "assignment omitted a row",
                    )
                })?;
                objective.add_product(self.gradient_left[i * k + j], b[column * k + other])?;
            }
        }
        objective.finish()
    }
    fn refine(
        &self,
        direction: QapDirection,
        p: &mut [usize],
        mut value: S,
        options: &QapOptions<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(S, usize, Stop), SciOxideError> {
        let mut checks = 0;
        let mut stop = Stop::Completed;
        'search: loop {
            let mut improved = false;
            for x in 0..self.free_a.len() {
                for y in x + 1..self.free_a.len() {
                    if checks >= options.max_swap_checks.get() {
                        stop = Stop::Method;
                        break 'search;
                    }
                    if !context.record_proposal()?.is_granted() {
                        stop = Stop::Budget(OptimizeTermination::ProposalLimit);
                        break 'search;
                    }
                    checks += 1;
                    let i = self.free_a[x];
                    let j = self.free_a[y];
                    let delta = self.swap_delta(p, i, j)?;
                    if better(delta, S::ZERO, direction)
                        && better(finite(value + delta)?, value, direction)
                    {
                        p.swap(i, j);
                        let checked = score(&self.a, &self.b, self.n, p)?;
                        if better(checked, value, direction) {
                            value = checked;
                            improved = true;
                            break;
                        }
                        p.swap(i, j);
                    }
                }
                if improved {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        Ok((value, checks, stop))
    }
    fn swap_delta(&self, p: &[usize], i: usize, j: usize) -> Result<S, SciOxideError> {
        let n = self.n;
        let u = p[i];
        let v = p[j];
        let mut sum = scioxide_core::ProductSum::<S>::new();
        for (k, &w) in p.iter().enumerate().take(n) {
            if k == i || k == j {
                continue;
            }
            for (a, new, old) in [
                (self.a[i * n + k], self.b[v * n + w], self.b[u * n + w]),
                (self.a[j * n + k], self.b[u * n + w], self.b[v * n + w]),
                (self.a[k * n + i], self.b[w * n + v], self.b[w * n + u]),
                (self.a[k * n + j], self.b[w * n + u], self.b[w * n + v]),
            ] {
                sum.add_product(a, new)?;
                sum.subtract_product(a, old)?;
            }
        }
        for (a, new, old) in [
            (self.a[i * n + i], self.b[v * n + v], self.b[u * n + u]),
            (self.a[j * n + j], self.b[u * n + u], self.b[v * n + v]),
            (self.a[i * n + j], self.b[v * n + u], self.b[u * n + v]),
            (self.a[j * n + i], self.b[u * n + v], self.b[v * n + u]),
        ] {
            sum.add_product(a, new)?;
            sum.subtract_product(a, old)?;
        }
        sum.finish()
    }
}
fn better<S: RealScalar>(a: S, b: S, direction: QapDirection) -> bool {
    match direction {
        QapDirection::Minimize => a < b,
        QapDirection::Maximize => a > b,
    }
}

/// Sets the heuristic, limits, restarts and execution policy of [`QapPlan`].
///
/// The storage cap `max_workspace_bytes` is checked on every call: a cap of zero
/// is `InvalidInput`, and a plan that needs more is `ResourceLimit`. See
/// [`QapResult::termination`] for the statuses and
/// [Termination](crate#termination) for the rules shared by all solvers.
#[derive(Clone, Debug)]
pub struct QapOptions<S: RealScalar> {
    method: QapMethod,
    max_workspace_bytes: usize,
    max_fw_iterations: Budget,
    max_swap_checks: Budget,
    step_tolerance: Option<StepTolerance<S>>,
    restarts: usize,
    seed: u64,
    execution: ExecutionPolicy,
}
impl<S: RealScalar> Default for QapOptions<S> {
    /// Uses FAQ followed by 2-opt, 30 FAQ iterations, 100,000 swap checks,
    /// step tolerance 0.03, no extra restarts, seed 42 and sequential execution.
    fn default() -> Self {
        Self {
            method: QapMethod::FaqThenTwoOpt,
            max_workspace_bytes: usize::MAX,
            max_fw_iterations: Budget::ONE.saturating_add(29),
            max_swap_checks: Budget::ONE.saturating_add(99_999),
            step_tolerance: None,
            restarts: 0,
            seed: 42,
            execution: ExecutionPolicy::Sequential,
        }
    }
}
impl<S: RealScalar> QapOptions<S> {
    /// Sets the assignment heuristic; the default is FAQ followed by 2-opt.
    pub const fn with_method(mut self, method: QapMethod) -> Self {
        self.method = method;
        self
    }
    /// Sets the workspace byte cap; the default `usize::MAX` imposes no limit.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Sets the maximum number of FAQ iterations per restart.
    pub const fn with_max_fw_iterations(mut self, value: Budget) -> Self {
        self.max_fw_iterations = value;
        self
    }
    /// Sets the maximum number of 2-opt swap checks per restart.
    pub const fn with_max_swap_checks(mut self, value: Budget) -> Self {
        self.max_swap_checks = value;
        self
    }
    /// Sets the tolerance on the normalized Frank–Wolfe step.
    pub const fn with_step_tolerance(mut self, value: StepTolerance<S>) -> Self {
        self.step_tolerance = Some(value);
        self
    }
    /// Sets the number of additional restarts from seeded random permutations.
    pub const fn with_restarts(mut self, value: usize) -> Self {
        self.restarts = value;
        self
    }
    /// Sets the seed of the restart permutations.
    pub const fn with_seed(mut self, value: u64) -> Self {
        self.seed = value;
        self
    }
    /// Normalized Frank-Wolfe step threshold; the default absolute threshold is 0.03.
    fn step_threshold(&self) -> Result<S, SciOxideError> {
        match self.step_tolerance {
            Some(tolerance) => tolerance.threshold(S::ONE),
            None => StepTolerance::new(number(0.03)?, S::ZERO)?.threshold(S::ONE),
        }
    }
    /// Sets whether restarts run sequentially (default) or in parallel through the executor of the execution context.
    pub const fn with_execution(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
}
/// Holds the reusable storage of [`QapPlan`] and the record of its last call.
#[derive(Debug)]
pub struct QapWorkspace<S: RealScalar> {
    shell: crate::attempt::Shell<()>,
    p: Vec<S>,
    q: Vec<S>,
    gradient: Vec<S>,
    products: Vec<S>,
    joined_products: Vec<S>,
    kernels: LinalgWorkspace<'static, S>,
    linear: Vec<S>,
    perm: Vec<usize>,
    candidate: Vec<usize>,
    best: Vec<usize>,
    assignment: AssignmentPlan,
    assignment_work: AssignmentWorkspace<S>,
}
impl<S: RealScalar> QapWorkspace<S> {
    /// Returns the record of the most recent call.
    pub const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.shell.last_attempt()
    }
}
/// Holds the result of [`QapPlan`].
///
/// After a completed solve, [`Self::permutation`] is a checked permutation and
/// [`Self::objective`] its objective recomputed from the original matrices.
#[derive(Debug)]
pub struct QapResult<S: RealScalar> {
    permutation: Vec<usize>,
    objective: Option<S>,
    relaxation_objective: Option<S>,
    fw_gap: Option<S>,
    fw_iterations: usize,
    swap_checks: usize,
    restart: usize,
    termination: OptimizeTermination,
}
impl<S: RealScalar> QapResult<S> {
    /// Creates an empty result for a problem of order `n`, for use with [`QapPlan::solve_into`].
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be allocated.
    pub fn empty(n: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            permutation: try_filled(n, 0, scioxide_core::quantity!("permutation"))?,
            objective: None,
            relaxation_objective: None,
            fw_gap: None,
            fw_iterations: 0,
            swap_checks: 0,
            restart: 0,
            termination: OptimizeTermination::NotStarted,
        })
    }
    /// Returns the best permutation `p`, with `p[i]` the node of `B` assigned to node `i` of `A`, if a restart completed.
    pub fn permutation(&self) -> Option<&[usize]> {
        self.objective.map(|_| self.permutation.as_slice())
    }
    /// Returns the objective of [`Self::permutation`].
    pub const fn objective(&self) -> Option<S> {
        self.objective
    }
    /// Returns the relaxed objective at the last FAQ iterate of the selected restart.
    pub const fn relaxation_objective(&self) -> Option<S> {
        self.relaxation_objective
    }
    /// Returns the Frank–Wolfe gap at the last FAQ iterate of the selected restart.
    ///
    /// The gap measures first-order stationarity of the relaxation; it does not
    /// bound the suboptimality of the permutation.
    pub const fn fw_gap(&self) -> Option<S> {
        self.fw_gap
    }
    /// Returns the total number of FAQ iterations over all restarts.
    pub const fn fw_iterations(&self) -> usize {
        self.fw_iterations
    }
    /// Returns the total number of 2-opt swaps checked over all restarts.
    pub const fn swap_checks(&self) -> usize {
        self.swap_checks
    }
    /// Returns the zero-based index of the restart that produced the result.
    pub const fn selected_restart(&self) -> usize {
        self.restart
    }
    /// Returns the reason the heuristic stopped.
    ///
    /// `Converged` means every selected stage finished. `IterationLimit` means
    /// an FAQ or swap-check limit, or the iteration limit of a parent solve
    /// context, was reached. `ProposalLimit`, `Cancelled` and `TimeLimit` report
    /// the parent limit or execution stop that ended the run. No status
    /// certifies a global optimum.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Independent original-coordinate O(n^4) definition, including seeded rows.
    fn direct_objective(a: &[f64], b: &[f64], x: &[f64], n: usize) -> f64 {
        let mut value = 0.0;
        for i in 0..n {
            for j in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        value += a[i * n + j] * b[r * n + s] * x[i * n + r] * x[j * n + s];
                    }
                }
            }
        }
        value
    }

    fn formulas<S: RealScalar>(symmetric: bool, fixed: &[(usize, usize)]) {
        let n = 4;
        let a: Vec<f64> = (0..n * n)
            .map(|i| ((i * 7 + 3) % 19) as f64 - 6.0)
            .collect();
        let mut b: Vec<f64> = (0..n * n)
            .map(|i| ((i * 3 + 2) % 13) as f64 - 4.0)
            .collect();
        let mut a = a;
        if symmetric {
            for i in 0..n {
                for j in 0..i {
                    a[i * n + j] = a[j * n + i];
                    b[i * n + j] = b[j * n + i];
                }
            }
        }
        let left: Vec<S> = a.iter().map(|&v| number(v).unwrap()).collect();
        let right: Vec<S> = b.iter().map(|&v| number(v).unwrap()).collect();
        let plan = QapProblemView::new(
            MatrixView::from_slice(n, n, &left).unwrap(),
            MatrixView::from_slice(n, n, &right).unwrap(),
            fixed,
        )
        .unwrap()
        .prepare()
        .unwrap();
        let options = QapOptions::default()
            .with_method(QapMethod::Faq)
            .with_max_workspace_bytes(usize::MAX);
        let mut w = plan.workspace(&options).unwrap();
        let k = plan.free_a.len();
        let mut x = alloc::vec![0.0; n * n];
        let mut endpoint = x.clone();
        for &(r, c) in fixed {
            x[r * n + c] = 1.0;
            endpoint[r * n + c] = 1.0;
        }
        let columns: Vec<_> = (0..k).map(|i| Some(k - 1 - i)).collect();
        for (i, &r) in plan.free_a.iter().enumerate() {
            for (j, &c) in plan.free_b.iter().enumerate() {
                let p = 0.5 / k as f64 + if i == j { 0.5 } else { 0.0 };
                w.p[i * k + j] = number(p).unwrap();
                x[r * n + c] = w.p[i * k + j].into_f64();
            }
            endpoint[r * n + plan.free_b[k - 1 - i]] = 1.0;
        }
        let constant = plan.fixed_terms(&mut w.linear).unwrap();
        let f0 = plan
            .gradient(
                &mut w,
                constant,
                ExecutionContext::sequential_native().into(),
            )
            .unwrap();
        let tolerance = if S::EPSILON.into_f64() > 1e-10 {
            2e-5
        } else {
            2e-12
        };
        let close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        };
        close(f0.into_f64(), direct_objective(&a, &b, &x, n));
        for (i, &r) in plan.free_a.iter().enumerate() {
            for (j, &c) in plan.free_b.iter().enumerate() {
                let mut expected = 0.0;
                for l in 0..n {
                    for s in 0..n {
                        expected += (a[r * n + l] * b[c * n + s] + a[l * n + r] * b[s * n + c])
                            * x[l * n + s];
                    }
                }
                close(w.gradient[i * k + j].into_f64(), expected);
            }
        }
        let f1 = plan
            .assignment_objective(&columns, &w.linear, constant)
            .unwrap();
        close(f1.into_f64(), direct_objective(&a, &b, &endpoint, n));
        let mut derivative = RealProductSum::<S>::new();
        for i in 0..k {
            for j in 0..k {
                let q = if j == k - 1 - i { S::ONE } else { S::ZERO };
                derivative
                    .add_product(w.gradient[i * k + j], q - w.p[i * k + j])
                    .unwrap();
            }
        }
        let derivative = derivative.finish().unwrap();
        let curvature = f1 - f0 - derivative;
        for step in [0.0, 0.125, 0.5, 0.875, 1.0] {
            let point: Vec<_> = x
                .iter()
                .zip(&endpoint)
                .map(|(&p, &q)| p + step * (q - p))
                .collect();
            close(
                line_value(f0, f1, derivative, curvature, number(step).unwrap())
                    .unwrap()
                    .into_f64(),
                direct_objective(&a, &b, &point, n),
            );
        }
        // Retain the independent permutation swap check from the original test.
        let mut permutation = alloc::vec![0; n];
        for &(r, c) in fixed {
            permutation[r] = c;
        }
        for (&r, &c) in plan.free_a.iter().zip(&plan.free_b) {
            permutation[r] = c;
        }
        let before = score(&left, &right, n, &permutation).unwrap();
        let delta = plan
            .swap_delta(&permutation, plan.free_a[0], plan.free_a[1])
            .unwrap();
        permutation.swap(plan.free_a[0], plan.free_a[1]);
        let after = score(&left, &right, n, &permutation).unwrap();
        close((after - before).into_f64(), delta.into_f64());
    }

    #[test]
    fn block_gradient_objective_and_line_match_original_model() {
        for symmetric in [false, true] {
            for fixed in [&[][..], &[(1, 3), (3, 0)][..]] {
                formulas::<f32>(symmetric, fixed);
                formulas::<f64>(symmetric, fixed);
            }
        }
    }

    #[test]
    fn line_search_selects_interior_endpoints_and_flat_steps() {
        for (f0, f1, derivative, curvature, expected) in [
            (1.0, 1.0, -4.0, 4.0, 0.5),
            (0.0, 3.0, 2.0, 1.0, 0.0),
            (4.0, 1.0, -4.0, 1.0, 1.0),
            (0.0, -1.0, -1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0, 0.0, 0.0),
            (2.0, 2.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 1.0, -2.0, 1.0),
        ] {
            let step = line_step(f0, f1, derivative, curvature, QapDirection::Minimize).unwrap();
            assert!((step - expected).abs() < 1e-14);
            let maximum =
                line_step(-f0, -f1, -derivative, -curvature, QapDirection::Maximize).unwrap();
            assert!((maximum - expected).abs() < 1e-14);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod termination_error_tests {
    use super::*;
    #[test]
    fn swap_limit_maps_to_iteration_limit() {
        let a = [0.0_f64, 3.0, 1.0, 8.0, 0.0, 2.0, 4.0, 6.0, 0.0];
        let b = [0.0, 5.0, 7.0, 2.0, 0.0, 9.0, 3.0, 1.0, 0.0];
        let plan = QapProblemView::new(
            MatrixView::from_slice(3, 3, &a).unwrap(),
            MatrixView::from_slice(3, 3, &b).unwrap(),
            &[],
        )
        .unwrap()
        .prepare()
        .unwrap();
        let options = QapOptions::default()
            .with_method(QapMethod::TwoOpt)
            .with_max_swap_checks(Budget::ONE);
        let result = plan
            .solve_with_options(QapDirection::Minimize, None, &options)
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        let counts = scioxide_core::SolveCounts::from_own(
            scioxide_core::EvaluationCounts::from_iterations(result.fw_iterations()),
        );
        let error = crate::result::termination_error(
            scioxide_core::argument!("a"),
            result.termination(),
            counts,
            None,
        )
        .unwrap();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
    }
}
