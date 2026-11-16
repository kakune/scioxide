//! Overlapping Schwarz preconditioners built from local solvers you provide.
use crate::helpers::invalid;
use alloc::vec::Vec;
use core::{mem::size_of, num::NonZeroUsize};
use scioxide_core::AlgebraicScalar;

use scioxide_core::backend::{
    checked_add as add, checked_mul as mul, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{
    Accuracy, Argument, Dimensions, EvaluationRole, ExecutionContext, LinearOperator,
    NonFiniteOrigin, NumericScalar, Preconditioner, RealScalar, SciOxideError, SolveContext,
    Tolerance,
};

use crate::SparseSolveOptions;

// Scalar slots of one subdomain's local storage.
fn local_span(m: usize, work_len: usize) -> Result<usize, SciOxideError> {
    let span = scioxide_core::quantity!("local_storage");
    add(mul(2, m, span)?, work_len, span)
}

/// Holds the global matrix `A` that multiplicative Schwarz uses to update the
/// residual.
///
/// With a sparse matrix, only the residual rows that a local correction changes
/// are recomputed. A matrix-free operator is applied to the whole vector after
/// every local correction.
///
/// # Examples
///
/// ```
/// use scioxide_core::LinearOperator;
/// use scioxide_sparse::{CsrMatrix, SchwarzOperatorView};
/// let matrix = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
/// let global = SchwarzOperatorView::Sparse(matrix.view());
/// assert_eq!(global.rows(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub enum SchwarzOperatorView<'a, S: NumericScalar> {
    /// A sparse matrix.
    Sparse(crate::SparseView<'a, S>),
    /// Any fixed linear operator.
    MatrixFree(&'a (dyn LinearOperator<S> + Sync)),
}
impl<S: NumericScalar> LinearOperator<S> for SchwarzOperatorView<'_, S> {
    fn rows(&self) -> usize {
        match self {
            Self::Sparse(m) => m.rows(),
            Self::MatrixFree(m) => m.rows(),
        }
    }
    fn cols(&self) -> usize {
        match self {
            Self::Sparse(m) => m.cols(),
            Self::MatrixFree(m) => m.cols(),
        }
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Sparse(m) => m.apply_workspace_len(),
            Self::MatrixFree(m) => m.apply_workspace_len(),
        }
    }
    fn apply_into(&self, x: &[S], y: &mut [S], w: &mut [S]) -> Result<(), SciOxideError> {
        match self {
            Self::Sparse(m) => m.apply_into(x, y, w),
            Self::MatrixFree(m) => m.apply_into(x, y, w),
        }
    }
}
/// Selects how a [`SchwarzPlan`] combines the local corrections.
///
/// Let `R_i` select the entries of subdomain `i`, `M_i` be its local solver and
/// `W_i` the diagonal matrix of its weights. For a residual `r` the
/// preconditioner returns `z`, where
///
/// ```text
/// additive:             z = Σᵢ R_iᵀ W_i M_i R_i r
/// restricted additive:  the same, with every weight 0 or 1
/// multiplicative:       z ← 0;  for i = 1, 2, …:  z ← z + R_iᵀ M_i R_i (r − A z)
/// ```
///
/// The additive kinds can solve the subdomains in parallel. Multiplicative
/// Schwarz visits them in the given order, one after another.
///
/// # Examples
///
/// All three kinds on tridiag(1, 3, 1) with subdomains {0, 1} and {1, 2}, whose
/// local blocks have the same inverse, exact in binary floating point:
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, SchwarzKind, SchwarzLocalSolveView, SchwarzOperatorView, SchwarzPlan,
///     SchwarzSubdomainView,
/// };
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![3.0_f64, 1., 1., 3., 1., 1., 3.],
/// )?;
/// let inverse = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![0.375_f64, -0.125, -0.125, 0.375],
/// )?; // inverse of [[3, 1], [1, 3]]
/// let solve = SchwarzLocalSolveView::InverseOperator(&inverse);
/// for (kind, weights, expected) in [
///     (
///         SchwarzKind::Additive,
///         Some([[1.0, 0.5], [0.5, 1.0]]),
///         [0.375, -0.0625, 0.0],
///     ),
///     (
///         SchwarzKind::RestrictedAdditive,
///         Some([[1.0, 1.0], [0.0, 1.0]]),
///         [0.375, -0.125, 0.0],
///     ),
///     (
///         SchwarzKind::Multiplicative,
///         None,
///         [0.375, -0.140625, 0.046875],
///     ),
/// ] {
///     let w = |i: usize| weights.as_ref().map(|w| &w[i][..]);
///     let domains = [
///         SchwarzSubdomainView {
///             indices: &[0, 1],
///             weights: w(0),
///             solve,
///         },
///         SchwarzSubdomainView {
///             indices: &[1, 2],
///             weights: w(1),
///             solve,
///         },
///     ];
///     let global = weights
///         .is_none()
///         .then_some(SchwarzOperatorView::Sparse(a.view()));
///     let plan = SchwarzPlan::new(3, kind, &domains, global)?;
///     assert_eq!(plan.apply(&[1.0, 0.0, 0.0])?, expected);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchwarzKind {
    /// Additive Schwarz with weights that sum to one at every index (a partition
    /// of unity).
    Additive,
    /// Restricted additive Schwarz: every index belongs to exactly one
    /// subdomain, which has weight 1 there; overlapping copies have weight 0.
    RestrictedAdditive,
    /// Multiplicative Schwarz: the residual is updated after each local
    /// correction. It needs the global matrix and no weights.
    Multiplicative,
}

/// Holds the solver `M_i` of one subdomain, which maps a local residual to a
/// local correction.
///
/// It follows the slice and workspace conventions of the core
/// [`LinearOperator`] and [`Preconditioner`] traits.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, JacobiPreconditioner, SchwarzKind, SchwarzLocalSolveView, SchwarzPlan,
///     SchwarzSubdomainView,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![2.0_f64, -1., -1., 4.],
/// )?;
/// let jacobi = JacobiPreconditioner::new(&a)?;
/// let domain = [SchwarzSubdomainView {
///     indices: &[0, 1],
///     weights: Some(&[1.0, 1.0]),
///     solve: SchwarzLocalSolveView::Preconditioner(&jacobi),
/// }];
/// let plan = SchwarzPlan::new(2, SchwarzKind::Additive, &domain, None)?;
/// assert_eq!(plan.apply(&[1.0, 1.0])?, [0.5, 0.25]);
/// // An inverse operator must be square.
/// let wide = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![0], vec![1.0_f64])?;
/// let domain = [SchwarzSubdomainView {
///     indices: &[0],
///     weights: Some(&[1.0]),
///     solve: SchwarzLocalSolveView::InverseOperator(&wide),
/// }];
/// let e = SchwarzPlan::new(1, SchwarzKind::Additive, &domain, None).err();
/// assert_eq!(
///     e.map(|e| e.kind()),
///     Some(SciOxideErrorKind::DimensionMismatch)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub enum SchwarzLocalSolveView<'a, S: NumericScalar> {
    /// An operator that applies the inverse of the local matrix, such as a
    /// [`SparseLu`](crate::SparseLu) factor.
    InverseOperator(&'a (dyn LinearOperator<S> + Sync)),
    /// A preconditioner, such as a [`JacobiPreconditioner`](crate::JacobiPreconditioner).
    Preconditioner(&'a (dyn Preconditioner<S> + Sync)),
}

impl<S: NumericScalar> SchwarzLocalSolveView<'_, S> {
    fn dimension(self) -> Result<usize, SciOxideError> {
        match self {
            Self::InverseOperator(op) if op.rows() == op.cols() => Ok(op.rows()),
            Self::InverseOperator(op) => Err(SciOxideError::dimension_mismatch(
                Dimensions::two(op.rows(), op.rows()),
                Dimensions::two(op.rows(), op.cols()),
            )),
            Self::Preconditioner(p) => Ok(p.dimension()),
        }
    }

    fn workspace_len(self) -> Result<usize, SciOxideError> {
        match self {
            Self::InverseOperator(op) => op.apply_workspace_len(),
            Self::Preconditioner(p) => p.precondition_workspace_len(),
        }
    }

    fn role(self) -> EvaluationRole {
        match self {
            Self::InverseOperator(_) => EvaluationRole::LinearSolve,
            Self::Preconditioner(_) => EvaluationRole::Preconditioner,
        }
    }

    fn apply(self, input: &[S], output: &mut [S], work: &mut [S]) -> Result<(), SciOxideError> {
        match self {
            Self::InverseOperator(op) => op.apply_into(input, output, work),
            Self::Preconditioner(p) => p.precondition_into(input, output, work),
        }
    }
}

/// Describes one subdomain of a [`SchwarzPlan`]: its global indices, its
/// weights and its local solver.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, SchwarzKind, SchwarzLocalSolveView, SchwarzPlan, SchwarzSubdomainView,
/// };
/// let inverse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0_f64])?;
/// let solve = SchwarzLocalSolveView::InverseOperator(&inverse);
/// // Two half-weighted copies of coordinate 0 form a partition of unity.
/// let halves = [SchwarzSubdomainView {
///     indices: &[0],
///     weights: Some(&[0.5]),
///     solve,
/// }; 2];
/// assert!(SchwarzPlan::new(1, SchwarzKind::Additive, &halves, None).is_ok());
/// // A single half-weighted copy does not sum to one.
/// let e = SchwarzPlan::new(1, SchwarzKind::Additive, &halves[..1], None).err();
/// assert_eq!(e.map(|e| e.kind()), Some(SciOxideErrorKind::InvalidInput));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct SchwarzSubdomainView<'a, S: NumericScalar> {
    /// The global indices of the subdomain, in strictly increasing order.
    pub indices: &'a [usize],
    /// One weight in `[0, 1]` per index for the additive kinds; `None` for
    /// multiplicative Schwarz.
    pub weights: Option<&'a [S::Real]>,
    /// The local solver.
    pub solve: SchwarzLocalSolveView<'a, S>,
}

/// Holds the weight-sum tolerance and the storage limits of a [`SchwarzPlan`].
///
/// The partition tolerance bounds how far the weights at each index may sum
/// from one for the additive kinds; the default is
/// [`Tolerance::from_machine_epsilon`]. The plan limit covers the index and
/// weight copies of the plan and the temporary storage of
/// [`SchwarzPlan::with_options`], and the workspace limit the scratch of one
/// application; both default to `usize::MAX` (no limit), because they are only
/// compared. The local solvers and allocator overhead are not counted.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, SchwarzKind, SchwarzLocalSolveView, SchwarzOptions, SchwarzPlan,
///     SchwarzSubdomainView,
/// };
/// let options = SchwarzOptions::<f64>::new()
///     .with_max_plan_bytes(4096)
///     .with_max_workspace_bytes(1024);
/// assert_eq!(
///     (options.max_plan_bytes(), options.max_workspace_bytes()),
///     (4096, 1024)
/// );
/// let inverse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![0.5_f64])?;
/// let domain = [SchwarzSubdomainView {
///     indices: &[0],
///     weights: Some(&[1.0]),
///     solve: SchwarzLocalSolveView::InverseOperator(&inverse),
/// }];
/// let plan =
///     |options| SchwarzPlan::with_options(1, SchwarzKind::Additive, &domain, None, options);
/// assert!(plan(options)?.plan_bytes() <= options.max_plan_bytes());
/// // A zero plan budget cannot hold even the subdomain records.
/// let e = plan(options.with_max_plan_bytes(0)).err();
/// assert_eq!(e.map(|e| e.kind()), Some(SciOxideErrorKind::ResourceLimit));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchwarzOptions<R: RealScalar> {
    partition_tolerance: Tolerance<R>,
    max_plan_bytes: usize,
    max_workspace_bytes: usize,
}

impl<R: RealScalar> Default for SchwarzOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RealScalar> SchwarzOptions<R> {
    /// Creates options with the partition tolerance
    /// [`Tolerance::from_machine_epsilon`] and no storage limits.
    pub fn new() -> Self {
        Self {
            partition_tolerance: Tolerance::from_machine_epsilon(),
            max_plan_bytes: usize::MAX,
            max_workspace_bytes: usize::MAX,
        }
    }
    /// Sets the tolerance within which the weights at each index must sum to
    /// one for the additive kinds.
    pub const fn with_partition_tolerance(mut self, value: Tolerance<R>) -> Self {
        self.partition_tolerance = value;
        self
    }
    /// Returns the partition tolerance.
    pub const fn partition_tolerance(self) -> Tolerance<R> {
        self.partition_tolerance
    }
    /// Sets the limit on plan storage, including temporary storage during
    /// construction.
    pub const fn with_max_plan_bytes(mut self, value: usize) -> Self {
        self.max_plan_bytes = value;
        self
    }
    /// Returns the limit on plan storage.
    pub const fn max_plan_bytes(self) -> usize {
        self.max_plan_bytes
    }
    /// Sets the limit on the workspace bytes of one application.
    /// [`SchwarzPlan::with_options`] fails if one right-hand side already
    /// needs more.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the limit on the workspace bytes of one application.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
}

#[derive(Clone)]
struct Subdomain<'a, S: NumericScalar> {
    indices: Vec<usize>,
    weights: Vec<S::Real>,
    solve: SchwarzLocalSolveView<'a, S>,
    work_len: usize,
    offset: usize,
    affected: Vec<usize>,
}

/// Applies an overlapping Schwarz preconditioner, as selected by
/// [`SchwarzKind`].
///
/// The plan copies the subdomain indices and weights and borrows the local
/// solvers and the global matrix, which must stay unchanged while it is used.
/// `apply_into` writes `z` for each right-hand side, and the plan implements the
/// core [`Preconditioner`] trait for Krylov solvers. Each right-hand side counts
/// as one iteration, each local solve as one evaluation, and for multiplicative
/// Schwarz each residual update as one operator evaluation. The additive kinds
/// distribute the subdomains over workers under the
/// [`ExecutionPolicy`](scioxide_core::ExecutionPolicy) of the call and add the
/// corrections in subdomain order, so the result does not depend on the number
/// of workers. Multiplicative Schwarz always runs sequentially. Sums use the
/// [`Accuracy`] of the call; local solvers use their own arithmetic.
///
/// # Examples
///
/// Additive Schwarz for tridiag(−1, 2, −1) with overlapping subdomains {0, 1}
/// and {1, 2}; both local blocks `[[2, −1], [−1, 2]]` share one sparse LU:
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, SchwarzKind, SchwarzLocalSolveView, SchwarzPlan, SchwarzSubdomainView, SparseLu,
/// };
/// let local =
///     CsrMatrix::from_triplets(2, 2, &[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0_f64, -1., -1., 2.])?;
/// let lu = SparseLu::new(&local)?;
/// let solve = SchwarzLocalSolveView::InverseOperator(&lu);
/// let domain = |indices, weights| SchwarzSubdomainView {
///     indices,
///     weights,
///     solve,
/// };
/// let domains = [
///     domain(&[0, 1], Some(&[1.0, 0.5])),
///     domain(&[1, 2], Some(&[0.5, 1.0])),
/// ];
/// let plan = SchwarzPlan::new(3, SchwarzKind::Additive, &domains, None)?;
/// let correction = plan.apply(&[1.0, 0.0, 0.0])?;
/// // The first local solve gives [2/3, 1/3]; the weights halve the shared entry.
/// let expected = [2.0 / 3.0, 1.0 / 6.0, 0.0];
/// assert!(
///     correction
///         .iter()
///         .zip(expected)
///         .all(|(c, e)| (c - e).abs() < 1e-15)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - B. Smith, P. Bjørstad and W. Gropp, *Domain Decomposition*, Cambridge
///   University Press, 1996.
/// - X.-C. Cai and M. Sarkis, "A restricted additive Schwarz preconditioner for
///   general sparse linear systems", *SIAM J. Sci. Comput.* 21 (1999), 792–797.
#[derive(Clone)]
pub struct SchwarzPlan<'a, S: NumericScalar> {
    dimension: usize,
    kind: SchwarzKind,
    domains: Vec<Subdomain<'a, S>>,
    operator: Option<SchwarzOperatorView<'a, S>>,
    operator_work_len: usize,
    plan_bytes: usize,
    preparation_peak_bytes: usize,
    workspace_len: usize,
    max_workspace_bytes: usize,
}

impl<'a, S: NumericScalar> SchwarzPlan<'a, S> {
    /// Creates a Schwarz preconditioner of order `dimension` from `subdomains`
    /// with the default [`SchwarzOptions`]: weights at each index sum to one
    /// within [`Tolerance::from_machine_epsilon`], and storage is not limited.
    ///
    /// This is [`with_options`](Self::with_options) with
    /// [`SchwarzOptions::new`], which states the rules for the subdomains.
    ///
    /// # Errors
    ///
    /// - The errors of [`with_options`](Self::with_options) other than
    ///   `ResourceLimit`.
    pub fn new(
        dimension: usize,
        kind: SchwarzKind,
        subdomains: &[SchwarzSubdomainView<'a, S>],
        operator: Option<SchwarzOperatorView<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(dimension, kind, subdomains, operator, SchwarzOptions::new())
    }
    /// Creates a Schwarz preconditioner of order `dimension` from `subdomains`,
    /// with the weight-sum tolerance `partition_tolerance` and the storage
    /// limits of `options`.
    ///
    /// For the additive kinds, the weights at each index must be positive in sum
    /// and sum to one within `options.partition_tolerance()`; for restricted
    /// additive Schwarz the sum must be exactly one. Multiplicative Schwarz needs
    /// `operator`, the square global matrix, and the additive kinds must not get
    /// one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a subdomain is empty or its indices are out of range
    ///   or not strictly increasing; if weights are missing for an additive kind
    ///   or given for multiplicative Schwarz; if a weight lies outside `[0, 1]`,
    ///   a restricted weight is not 0 or 1, or the weights at some index do not
    ///   sum to one; if `operator` is missing for multiplicative Schwarz or given
    ///   for an additive kind; or if there are no subdomains for a nonzero
    ///   `dimension`, or any for `dimension` 0.
    /// - `DimensionMismatch` if a local solver, weight slice or `operator` has
    ///   the wrong size.
    /// - `NonFinite` if a weight is NaN or infinite.
    /// - `ResourceLimit` if the plan or one application's workspace exceeds
    ///   the byte limits of `options`.
    /// - `SizeOverflow`, `AllocationFailure` if storage cannot be sized or
    ///   allocated.
    pub fn with_options(
        dimension: usize,
        kind: SchwarzKind,
        subdomains: &[SchwarzSubdomainView<'a, S>],
        operator: Option<SchwarzOperatorView<'a, S>>,
        options: SchwarzOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        const OPERATOR: Argument = scioxide_core::argument!("operator");
        const SUBDOMAINS: Argument = scioxide_core::argument!("subdomains");
        const WEIGHTS: Argument = scioxide_core::argument!("subdomains.weights");
        let plan = scioxide_core::quantity!("plan_bytes");
        let peak = scioxide_core::quantity!("preparation_peak_bytes");
        let slots = scioxide_core::quantity!("workspace_len");
        if kind == SchwarzKind::Multiplicative {
            if dimension > 0 {
                let op = operator.ok_or_else(|| {
                    invalid(OPERATOR, "multiplicative Schwarz needs a global operator")
                })?;
                crate::helpers::shape((op.rows(), op.cols()), (dimension, dimension))?;
            } else if let Some(op) = operator {
                crate::helpers::shape((op.rows(), op.cols()), (0, 0))?;
            }
        } else if operator.is_some() {
            return Err(invalid(
                OPERATOR,
                "additive Schwarz does not use a global operator",
            ));
        }
        if dimension == 0 {
            if !subdomains.is_empty() {
                return Err(invalid(
                    SUBDOMAINS,
                    "empty global dimension requires no subdomains",
                ));
            }
        } else if subdomains.is_empty() {
            return Err(invalid(
                SUBDOMAINS,
                "subdomains must cover the global dimension",
            ));
        }
        let mut plan_bytes = 0usize;
        let mut domain_slots = 0usize;
        let mut max_local_slots = 0usize;
        let provider_bytes = mul(subdomains.len(), size_of::<usize>(), peak)?;
        if provider_bytes > options.max_plan_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("plan_bytes"),
                provider_bytes,
                options.max_plan_bytes,
            ));
        }
        let mut provider_work = try_with_capacity(
            subdomains.len(),
            scioxide_core::quantity!("provider_workspace_lengths"),
        )?;
        for domain in subdomains {
            let m = domain.indices.len();
            if m == 0 {
                return Err(invalid(
                    scioxide_core::argument!("subdomains.indices"),
                    "empty subdomain",
                ));
            }
            if domain.indices.iter().any(|&i| i >= dimension)
                || domain.indices.windows(2).any(|p| p[0] >= p[1])
            {
                return Err(invalid(
                    scioxide_core::argument!("subdomains.indices"),
                    "subdomain indices must increase and be in range",
                ));
            }
            crate::helpers::exact(domain.solve.dimension()?, m)?;
            let weights = match (kind, domain.weights) {
                (SchwarzKind::Multiplicative, None) => &[][..],
                (SchwarzKind::Multiplicative, Some(_)) => {
                    return Err(invalid(
                        WEIGHTS,
                        "multiplicative Schwarz does not use weights",
                    ));
                }
                (_, Some(w)) => {
                    crate::helpers::exact(w.len(), m)?;
                    w
                }
                _ => {
                    return Err(invalid(
                        WEIGHTS,
                        "additive weights must match the local map",
                    ));
                }
            };
            if weights.iter().any(|&w| !w.is_finite()) {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "weights must be finite",
                ));
            }
            if weights
                .iter()
                .any(|&w| w < S::Real::ZERO || w > S::Real::ONE)
            {
                return Err(invalid(WEIGHTS, "weights must be in [0, 1]"));
            }
            if kind == SchwarzKind::RestrictedAdditive
                && weights
                    .iter()
                    .any(|&w| w != S::Real::ZERO && w != S::Real::ONE)
            {
                return Err(invalid(WEIGHTS, "restricted weights must be zero or one"));
            }
            let provider = domain.solve.workspace_len()?;
            provider_work.push(provider);
            let local_slots = local_span(m, provider)?;
            domain_slots = add(domain_slots, local_slots, slots)?;
            max_local_slots = max_local_slots.max(local_slots);
            plan_bytes = add(plan_bytes, mul(m, size_of::<usize>(), plan)?, plan)?;
            plan_bytes = add(
                plan_bytes,
                mul(weights.len(), size_of::<S::Real>(), plan)?,
                plan,
            )?;
        }
        let mut affected_maps = Vec::new();
        let mut affected_markers = Vec::new();
        let mut map_preparation_bytes = 0;
        if let Some(SchwarzOperatorView::Sparse(matrix)) = operator {
            let marker_bytes = mul(dimension, size_of::<usize>(), peak)?;
            let map_records = mul(subdomains.len(), size_of::<Vec<usize>>(), peak)?;
            map_preparation_bytes = add(marker_bytes, map_records, peak)?;
            let initial = add(
                add(plan_bytes, provider_bytes, peak)?,
                map_preparation_bytes,
                peak,
            )?;
            if initial > options.max_plan_bytes {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("plan_bytes"),
                    initial,
                    options.max_plan_bytes,
                ));
            }
            affected_markers = try_filled(
                dimension,
                usize::MAX,
                scioxide_core::quantity!("affected_row_markers"),
            )?;
            let marker = &mut affected_markers;
            affected_maps =
                try_with_capacity(subdomains.len(), scioxide_core::quantity!("affected_rows"))?;
            for (id, domain) in subdomains.iter().enumerate() {
                let mut count = 0;
                if id + 1 < subdomains.len() {
                    for &col in domain.indices {
                        for (r, _, _) in matrix.column(col) {
                            if marker[r] != id {
                                marker[r] = id;
                                count += 1;
                            }
                        }
                    }
                }
                plan_bytes = add(plan_bytes, mul(count, size_of::<usize>(), plan)?, plan)?;
                let required = add(
                    add(plan_bytes, provider_bytes, peak)?,
                    map_preparation_bytes,
                    peak,
                )?;
                if required > options.max_plan_bytes {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("plan_bytes"),
                        required,
                        options.max_plan_bytes,
                    ));
                }
                let mut rows = try_with_capacity(count, scioxide_core::quantity!("affected_rows"))?;
                if count > 0 {
                    for &col in domain.indices {
                        for (r, _, _) in matrix.column(col) {
                            if marker[r] == id {
                                rows.push(r);
                                marker[r] = usize::MAX;
                            }
                        }
                    }
                }
                rows.sort_unstable();
                affected_maps.push(rows);
            }
        }
        plan_bytes = add(
            plan_bytes,
            mul(subdomains.len(), size_of::<Subdomain<'a, S>>(), plan)?,
            plan,
        )?;
        let preparation_peak_bytes = if kind == SchwarzKind::Multiplicative {
            add(
                add(plan_bytes, provider_bytes, peak)?,
                map_preparation_bytes,
                peak,
            )?
        } else {
            add(
                add(plan_bytes, provider_bytes, peak)?,
                mul(dimension, size_of::<S::Real>(), peak)?,
                peak,
            )?
        };
        if preparation_peak_bytes > options.max_plan_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("plan_bytes"),
                preparation_peak_bytes,
                options.max_plan_bytes,
            ));
        }
        let matrix_free = matches!(operator, Some(SchwarzOperatorView::MatrixFree(_)));
        let operator_work_len = match operator {
            Some(SchwarzOperatorView::MatrixFree(op)) => op.apply_workspace_len()?,
            _ => 0,
        };
        let workspace_len = if kind == SchwarzKind::Multiplicative {
            add(
                add(
                    mul(if matrix_free { 3 } else { 2 }, dimension, slots)?,
                    max_local_slots,
                    slots,
                )?,
                operator_work_len,
                slots,
            )?
        } else {
            add(dimension, domain_slots, slots)?
        };
        let workspace_bytes = mul(
            workspace_len,
            size_of::<S>(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if workspace_bytes > options.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                workspace_bytes,
                options.max_workspace_bytes,
            ));
        }
        let mut sums = if kind == SchwarzKind::Multiplicative {
            Vec::new()
        } else {
            try_filled(
                dimension,
                S::Real::ZERO,
                scioxide_core::quantity!("weight_sums"),
            )?
        };
        let mut domains = try_with_capacity(
            subdomains.len(),
            scioxide_core::quantity!("subdomain_records"),
        )?;
        let mut offset = 0;
        let mut affected = affected_maps.into_iter();
        for (domain, &work_len) in subdomains.iter().zip(&provider_work) {
            let m = domain.indices.len();
            let weights = domain.weights.unwrap_or(&[]);
            for (&index, &weight) in domain.indices.iter().zip(weights) {
                sums[index] += weight;
            }
            domains.push(Subdomain {
                indices: try_copy_slice(
                    domain.indices,
                    scioxide_core::quantity!("subdomain_indices"),
                )?,
                weights: try_copy_slice(weights, scioxide_core::quantity!("subdomain_weights"))?,
                solve: domain.solve,
                work_len,
                offset,
                affected: affected.next().unwrap_or_default(),
            });
            offset = add(offset, local_span(m, work_len)?, slots)?;
        }
        drop(affected_markers);
        if kind != SchwarzKind::Multiplicative {
            let threshold = options.partition_tolerance.threshold(S::Real::ONE)?;
            if sums.iter().any(|&sum| {
                !sum.is_finite()
                    || sum <= S::Real::ZERO
                    || (kind == SchwarzKind::RestrictedAdditive && sum != S::Real::ONE)
                    || !scioxide_core::backend::absolute_difference_le(sum, S::Real::ONE, threshold)
            }) {
                return Err(invalid(
                    WEIGHTS,
                    "injection weights must form a partition of unity",
                ));
            }
        }
        Ok(Self {
            dimension,
            kind,
            domains,
            operator,
            operator_work_len,
            plan_bytes,
            preparation_peak_bytes,
            workspace_len,
            max_workspace_bytes: options.max_workspace_bytes,
        })
    }

    /// Returns the bytes the plan holds: indices, weights and, for a sparse
    /// global matrix, the rows each subdomain affects.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }

    /// Returns the peak bytes used while the plan was created.
    pub const fn preparation_peak_bytes(&self) -> usize {
        self.preparation_peak_bytes
    }

    /// Returns the kind of Schwarz preconditioner.
    pub const fn kind(&self) -> SchwarzKind {
        self.kind
    }

    // One local correction.
    fn local(&self, index: usize, input: &[S], storage: &mut [S]) -> Result<(), SciOxideError> {
        let domain = &self.domains[index];
        let m = domain.indices.len();
        let (gathered, tail) = storage.split_at_mut(m);
        let (correction, work) = tail.split_at_mut(m);
        for (&global, local) in domain.indices.iter().zip(gathered.iter_mut()) {
            *local = input[global];
        }
        domain
            .solve
            .apply(gathered, correction, &mut work[..domain.work_len])?;
        if !correction.iter().all(|x| x.has_finite_components()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Callback,
                "local solve returned non-finite correction",
            ));
        }
        Ok(())
    }

    // Weighted combination in subdomain order.
    fn combine(
        &self,
        staged: &mut [S],
        local_storage: &[S],
        accuracy: Accuracy,
    ) -> Result<(), SciOxideError> {
        for domain in &self.domains {
            let m = domain.indices.len();
            let correction = &local_storage[domain.offset + m..domain.offset + 2 * m];
            for ((&global, &weight), &value) in
                domain.indices.iter().zip(&domain.weights).zip(correction)
            {
                staged[global] = scioxide_core::backend::standard::sum_products(
                    [(staged[global], S::ONE), (value, S::from_real(weight))].into_iter(),
                    accuracy,
                )?;
            }
        }
        Ok(())
    }

    // Additive application: local solves follow the dispatch policy and their
    // corrections always combine in subdomain order.
    fn additive(
        &self,
        input: &[S],
        staged: &mut [S],
        local_storage: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        staged.fill(S::ZERO);
        let plan = execution.plan(
            options.execution_policy(),
            self.domains.len(),
            scioxide_core::PlanLimits {
                work_per_item: NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: crate::execution::policy_workers(options.execution_policy()),
            },
        );
        let total = local_storage.len();
        let mut remaining = &mut local_storage[..];
        let inputs = plan.partitions().map(|partition| {
            let range = partition.range();
            let start = self.domains[range.start].offset;
            let end = self
                .domains
                .get(range.end)
                .map_or(total, |next| next.offset);
            let (part, tail) = core::mem::take(&mut remaining).split_at_mut(end - start);
            remaining = tail;
            part
        });
        crate::execution::dispatch(execution, plan, inputs, &|partition, part, scope| {
            let range = partition.range();
            let base = self.domains[range.start].offset;
            for i in range {
                scope.check()?;
                let domain = &self.domains[i];
                let span = local_span(domain.indices.len(), domain.work_len)?;
                let offset = domain.offset - base;
                self.local(i, input, &mut part[offset..offset + span])?;
            }
            Ok(())
        })?;
        execution.check()?;
        self.combine(staged, local_storage, options.accuracy())
    }

    // Ordered multiplicative application into `candidate`.
    fn multiplicative(
        &self,
        input: &[S],
        candidate: &mut [S],
        work: &mut [S],
        accuracy: Accuracy,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.dimension;
        let (residual, rest) = work.split_at_mut(n);
        let image_len = if matches!(self.operator, Some(SchwarzOperatorView::MatrixFree(_))) {
            n
        } else {
            0
        };
        let (image, rest) = rest.split_at_mut(image_len);
        candidate.fill(S::ZERO);
        residual.copy_from_slice(input);
        if self.domains.is_empty() {
            return Ok(());
        }
        let op = self
            .operator
            .ok_or_else(|| invalid(scioxide_core::argument!("self"), "missing global operator"))?;
        let global_work = self.operator_work_len;
        for i in 0..self.domains.len() {
            execution.check()?;
            let domain = &self.domains[i];
            let m = domain.indices.len();
            let local_len = local_span(m, domain.work_len)?;
            let (local, global_work_storage) = rest.split_at_mut(local_len);
            self.local(i, residual, local)?;
            let correction = &local[m..2 * m];
            for (&global, &value) in domain.indices.iter().zip(correction) {
                candidate[global] = scioxide_core::backend::standard::sum_products(
                    [(candidate[global], S::ONE), (value, S::ONE)].into_iter(),
                    accuracy,
                )?;
            }
            if i + 1 < self.domains.len() {
                match op {
                    SchwarzOperatorView::Sparse(matrix) => {
                        for &row in &domain.affected {
                            let ax = matrix.dot(row, candidate, (false, false), accuracy)?;
                            residual[row] = scioxide_core::backend::standard::sum_products(
                                [(input[row], S::ONE), (-ax, S::ONE)].into_iter(),
                                accuracy,
                            )?;
                        }
                    }
                    SchwarzOperatorView::MatrixFree(operator) => {
                        operator.apply_into(
                            candidate,
                            image,
                            &mut global_work_storage[..global_work],
                        )?;
                        if !image.iter().all(|x| x.has_finite_components()) {
                            return Err(SciOxideError::non_finite(
                                NonFiniteOrigin::Callback,
                                "global operator returned non-finite values",
                            ));
                        }
                        for ((r, &b), &ax) in residual.iter_mut().zip(input).zip(image.iter()) {
                            *r = scioxide_core::backend::standard::sum_products(
                                [(b, S::ONE), (-ax, S::ONE)].into_iter(),
                                accuracy,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Each right-hand side counts as one iteration, one provider evaluation per
/// local solve and, for multiplicative composition, one `LinearOperator`
/// evaluation per residual update. Additive local solves follow the options'
/// dispatch policy; multiplicative composition is sequential under every
/// policy. Products use the options' accuracy; injected providers keep their own
/// arithmetic.
impl<S: NumericScalar> crate::factor_solve::Application<S> for SchwarzPlan<'_, S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.workspace_len - self.dimension)
    }
    fn staging_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((0, self.max_workspace_bytes))
    }
    fn admit(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        crate::helpers::admit(context)?;
        let residuals = if self.kind == SchwarzKind::Multiplicative {
            self.domains.len().saturating_sub(1)
        } else {
            0
        };
        self.domains
            .iter()
            .map(|domain| domain.solve.role())
            .chain(core::iter::repeat_n(
                EvaluationRole::LinearOperator,
                residuals,
            ))
            .try_for_each(|role| {
                if context.record_provider(role)?.is_granted() {
                    Ok(())
                } else {
                    Err(crate::helpers::exhausted(
                        context,
                        scioxide_core::ConvergenceReason::EvaluationLimit,
                    ))
                }
            })
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        work: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.kind == SchwarzKind::Multiplicative {
            self.multiplicative(rhs, candidate, work, options.accuracy(), execution)
        } else {
            self.additive(rhs, candidate, work, options, execution)
        }
    }
}
crate::factor_solve::impl_applications!(SchwarzPlan<'_, S>);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use scioxide_core::AlgebraicScalar;
    #[cfg(feature = "std")]
    use scioxide_core::ExecutionPolicy;
    use scioxide_core::{Complex, ExecutionContext, Preconditioner, RealScalar};

    use super::*;
    use crate::{CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic};

    fn scalar<S: NumericScalar>(value: f64) -> S {
        S::from_real(S::Real::try_from_f64_lossy(value).unwrap())
    }

    fn limits<R: RealScalar>() -> SchwarzOptions<R> {
        SchwarzOptions::new()
            .with_max_plan_bytes(4096)
            .with_max_workspace_bytes(4096)
    }

    struct FailingSolve;
    impl<S: NumericScalar> Preconditioner<S> for FailingSolve {
        fn dimension(&self) -> usize {
            1
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            _: &[S],
            _: &mut [S],
            _: &mut [S],
        ) -> Result<(), SciOxideError> {
            Err(invalid(
                scioxide_core::argument!("input"),
                "local solve failed",
            ))
        }
    }

    fn numerical_families<S: NumericScalar>() {
        let local = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![scalar::<S>(2.), scalar(-1.), scalar(-1.), scalar(2.)],
        )
        .unwrap();
        let symbolic = SparseLuSymbolic::analyze(
            local.structural_pattern(),
            crate::SparseOrdering::default(),
            SparseLuLimits::new(4, 4096),
        )
        .unwrap();
        let inverse = SparseLu::factor(symbolic, local.view()).unwrap();
        let global = CsrMatrix::from_raw(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![
                scalar::<S>(2.),
                scalar(-1.),
                scalar(-1.),
                scalar(2.),
                scalar(-1.),
                scalar(-1.),
                scalar(2.),
            ],
        )
        .unwrap();
        let half = S::Real::try_from_f64_exact(0.5).unwrap();
        let one = S::Real::ONE;
        let zero = S::Real::ZERO;
        let weights_a = [one, half];
        let weights_b = [half, one];
        let ras_a = [one, one];
        let ras_b = [zero, one];
        let solve = SchwarzLocalSolveView::InverseOperator(&inverse);
        let rhs = [scalar::<S>(1.), S::ZERO, S::ZERO];
        for (kind, a, b, expected) in [
            (
                SchwarzKind::Additive,
                Some(&weights_a[..]),
                Some(&weights_b[..]),
                [2. / 3., 1. / 6., 0.],
            ),
            (
                SchwarzKind::RestrictedAdditive,
                Some(&ras_a[..]),
                Some(&ras_b[..]),
                [2. / 3., 1. / 3., 0.],
            ),
            (
                SchwarzKind::Multiplicative,
                None,
                None,
                [2. / 3., 4. / 9., 2. / 9.],
            ),
        ] {
            let maps = [
                SchwarzSubdomainView {
                    indices: &[0, 1],
                    weights: a,
                    solve,
                },
                SchwarzSubdomainView {
                    indices: &[1, 2],
                    weights: b,
                    solve,
                },
            ];
            let plan = SchwarzPlan::with_options(
                3,
                kind,
                &maps,
                if kind == SchwarzKind::Multiplicative {
                    Some(SchwarzOperatorView::MatrixFree(&global))
                } else {
                    None
                },
                limits(),
            )
            .unwrap();
            let mut output = [scalar::<S>(99.); 3];
            let mut workspace = vec![S::ZERO; plan.precondition_workspace_len().unwrap()];
            plan.precondition_into(&rhs, &mut output, &mut workspace)
                .unwrap();
            let epsilon = S::Real::EPSILON.try_to_f64_exact().unwrap();
            for (&actual, target) in output.iter().zip(expected) {
                assert!(
                    (actual - scalar::<S>(target))
                        .abs()
                        .try_to_f64_exact()
                        .unwrap()
                        < 20. * epsilon,
                    "{kind:?}: {actual:?} != {target:?}"
                );
            }
            let mut image = [S::ZERO; 3];
            let mut global_work = [S::ZERO; 3];
            global
                .apply_into(&output, &mut image, &mut global_work)
                .unwrap();
            let residual_norm = rhs
                .iter()
                .zip(image.iter())
                .map(|(&b, &ax)| {
                    let value = (b - ax).abs().try_to_f64_exact().unwrap();
                    value * value
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                residual_norm < 0.5,
                "{kind:?} did not reduce the original residual"
            );
            // Every policy publishes the same correction; multiplicative
            // composition runs sequentially under a parallel request.
            let workers = NonZeroUsize::new(2).unwrap();
            let options = crate::SparseSolveOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers));
            #[cfg(feature = "std")]
            let executor = scioxide_core::ThreadExecutor::new(workers);
            #[cfg(feature = "std")]
            let context = ExecutionContext::new(&executor);
            #[cfg(not(feature = "std"))]
            let context = ExecutionContext::sequential();
            let mut parallel = [S::ZERO; 3];
            let mut owned = plan.workspace(2).unwrap();
            plan.apply_into(&rhs, options, &mut parallel, &mut owned, context)
                .unwrap();
            assert_eq!(parallel, output);
            let block_rhs = [rhs, rhs].concat();
            let mut block = [S::ZERO; 6];
            plan.apply_block_into(
                scioxide_core::BlockVectorView::new(&block_rhs, 3, 2).unwrap(),
                options,
                &mut scioxide_core::BlockVectorViewMut::new(&mut block, 3, 2).unwrap(),
                &mut owned,
                context,
            )
            .unwrap();
            assert_eq!(block[..3], output);
            assert_eq!(block[3..], output);
        }
    }

    #[test]
    fn all_scalar_domains_and_methods() {
        numerical_families::<f32>();
        numerical_families::<f64>();
        numerical_families::<Complex<f32>>();
        numerical_families::<Complex<f64>>();
    }

    #[test]
    fn invalid_maps_budgets_and_failed_solve_preserve_output() {
        let failing = FailingSolve;
        let domain = SchwarzSubdomainView {
            indices: &[0],
            weights: Some(&[1.]),
            solve: SchwarzLocalSolveView::Preconditioner(&failing),
        };
        assert!(
            SchwarzPlan::with_options(
                1,
                SchwarzKind::Additive,
                &[domain],
                None,
                SchwarzOptions::new()
                    .with_max_plan_bytes(0)
                    .with_max_workspace_bytes(0)
            )
            .is_err()
        );
        let plan =
            SchwarzPlan::with_options(1, SchwarzKind::Additive, &[domain], None, limits()).unwrap();
        let mut output = [7.];
        let mut workspace = vec![0.; plan.precondition_workspace_len().unwrap()];
        assert!(
            plan.precondition_into(&[1.], &mut output, &mut workspace)
                .is_err()
        );
        assert_eq!(output, [7.]);
        assert!(plan.precondition_into(&[1.], &mut output, &mut []).is_err());
        assert_eq!(output, [7.]);
        let bad: SchwarzSubdomainView<'_, f64> = SchwarzSubdomainView {
            indices: &[0, 0],
            weights: Some(&[0.5, 0.5]),
            solve: SchwarzLocalSolveView::Preconditioner(&failing),
        };
        assert!(
            SchwarzPlan::with_options(1, SchwarzKind::Additive, &[bad], None, limits()).is_err()
        );
    }

    #[test]
    fn partition_and_empty_edges() {
        let identity = crate::IdentityPreconditioner::new(1);
        let solve: SchwarzLocalSolveView<'_, f64> =
            SchwarzLocalSolveView::Preconditioner(&identity);
        for weights in [[0.5, 0.25], [1.0, 1.0], [0.0, 0.0]] {
            let domains = [
                SchwarzSubdomainView {
                    indices: &[0],
                    weights: Some(&weights[..1]),
                    solve,
                },
                SchwarzSubdomainView {
                    indices: &[0],
                    weights: Some(&weights[1..]),
                    solve,
                },
            ];
            assert!(
                SchwarzPlan::with_options(1, SchwarzKind::Additive, &domains, None, limits())
                    .is_err()
            );
        }
        let weighted = [0.5];
        let domains = [
            SchwarzSubdomainView {
                indices: &[0],
                weights: Some(&weighted),
                solve,
            },
            SchwarzSubdomainView {
                indices: &[0],
                weights: Some(&weighted),
                solve,
            },
        ];
        assert!(
            SchwarzPlan::with_options(1, SchwarzKind::RestrictedAdditive, &domains, None, limits())
                .is_err()
        );
        let empty = SchwarzPlan::<f64>::with_options(0, SchwarzKind::Additive, &[], None, limits())
            .unwrap();
        empty.precondition_into(&[], &mut [], &mut []).unwrap();
        let empty_multiplicative =
            SchwarzPlan::<f64>::with_options(0, SchwarzKind::Multiplicative, &[], None, limits())
                .unwrap();
        empty_multiplicative
            .precondition_into(&[], &mut [], &mut [])
            .unwrap();
    }

    #[test]
    fn context_budget_and_linear_storage() {
        use scioxide_core::{
            Budget, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
        };
        let n = 1024;
        let identity = crate::IdentityPreconditioner::new(n);
        let indices: Vec<_> = (0..n).collect();
        let weights = vec![1.; n];
        let domains = [SchwarzSubdomainView {
            indices: &indices,
            weights: Some(&weights),
            solve: SchwarzLocalSolveView::Preconditioner(&identity),
        }];
        let plan = SchwarzPlan::with_options(
            n,
            SchwarzKind::Additive,
            &domains,
            None,
            SchwarzOptions::new()
                .with_max_plan_bytes(32 * n)
                .with_max_workspace_bytes(32 * n),
        )
        .unwrap();
        assert!(plan.plan_bytes() <= 17 * n);
        let len = plan.precondition_workspace_len().unwrap();
        assert!(len <= 4 * n);
        let mut workspace = vec![0.; len];
        let input = vec![1.; n];
        let mut output = vec![7.; n];
        // One iteration per right-hand side: a two-column block exceeds a local
        // allowance of one before any column runs.
        let limited = crate::SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
        // Owned staging counts against the byte limit; borrowed staging is checked
        // only by length.
        assert!(matches!(
            plan.workspace(2).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
        let mut staging = vec![0.; 2 * len];
        let mut owned = crate::SparseSolveWorkspace::borrowed(&mut staging);
        let block_input = vec![1.; 2 * n];
        let mut block_output = vec![7.; 2 * n];
        let error = plan
            .apply_block_into(
                scioxide_core::BlockVectorView::new(&block_input, n, 2).unwrap(),
                limited,
                &mut scioxide_core::BlockVectorViewMut::new(&mut block_output, n, 2).unwrap(),
                &mut owned,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
        assert!(block_output.iter().all(|&x| x == 7.));
        let evaluation = crate::SparseSolveOptions::new().with_budgets(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        );
        assert!(
            plan.apply_block_into(
                scioxide_core::BlockVectorView::new(&block_input, n, 2).unwrap(),
                evaluation,
                &mut scioxide_core::BlockVectorViewMut::new(&mut block_output, n, 2).unwrap(),
                &mut owned,
                ExecutionContext::sequential(),
            )
            .is_err()
        );
        // A child scope records the local solve in the parent's nested counts.
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(491),
            scioxide_core::algorithm_id!("sparse", "schwarz", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        plan.apply_into(
            &input,
            crate::SparseSolveOptions::new(),
            &mut output,
            &mut owned,
            &mut context,
        )
        .unwrap();
        let counts = owned.last_attempt().unwrap().counts();
        assert_eq!(counts.own().iterations(), 1);
        assert_eq!(counts.own().provider(EvaluationRole::Preconditioner), 1);
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::Preconditioner),
            1
        );
        output.fill(7.);
        plan.precondition_into(&input, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn singular_local_lu_rejects_before_plan() {
        let singular = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 2.0, 4.0],
        )
        .unwrap();
        let symbolic = SparseLuSymbolic::analyze(
            singular.structural_pattern(),
            crate::SparseOrdering::default(),
            SparseLuLimits::new(4, 4096),
        )
        .unwrap();
        assert!(SparseLu::factor(symbolic, singular.view()).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_local_failure_does_not_publish_partial_output() {
        let identity = crate::IdentityPreconditioner::new(1);
        let failing = FailingSolve;
        let one = [1.];
        let domains = [
            SchwarzSubdomainView {
                indices: &[0],
                weights: Some(&one),
                solve: SchwarzLocalSolveView::Preconditioner(&identity),
            },
            SchwarzSubdomainView {
                indices: &[1],
                weights: Some(&one),
                solve: SchwarzLocalSolveView::Preconditioner(&failing),
            },
        ];
        let plan =
            SchwarzPlan::with_options(2, SchwarzKind::Additive, &domains, None, limits()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut output = [7.; 2];
        let workers = NonZeroUsize::new(2).unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        assert!(
            plan.apply_into(
                &[1., 1.],
                crate::SparseSolveOptions::new()
                    .with_execution_policy(ExecutionPolicy::Parallel(workers)),
                &mut output,
                &mut workspace,
                ExecutionContext::new(&executor)
            )
            .is_err()
        );
        assert_eq!(output, [7.; 2]);
    }
}

#[cfg(test)]
mod locality_tests {
    use super::*;
    use crate::{DuplicatePolicy, ExplicitZeroPolicy, IdentityPreconditioner};
    fn compare<S: NumericScalar>() {
        let scalar = |r: f64, i: f64| {
            S::from_components(
                S::Real::try_from_f64_lossy(r).unwrap(),
                S::Real::try_from_f64_lossy(i).unwrap(),
            )
        };
        let n = 9usize;
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in i.saturating_sub(1)..(i + 2).min(n) {
                rows.push(i);
                columns.push(j);
                values.push(if i == j {
                    scalar(3.5, 0.)
                } else {
                    scalar(-0.25, 0.125)
                });
            }
        }
        let matrix = {
            let mut builder = crate::SparseBuilder::new(n, n);
            builder.extend(&rows, &columns, &values).and_then(|()| {
                builder.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            })
        }
        .unwrap()
        .to_csr()
        .unwrap();
        let local = IdentityPreconditioner::new(1);
        let indices: Vec<_> = (0..n).map(|i| [i]).collect();
        let domains: Vec<_> = indices
            .iter()
            .map(|indices| SchwarzSubdomainView {
                indices,
                weights: None,
                solve: SchwarzLocalSolveView::Preconditioner(&local),
            })
            .collect();
        let limits = SchwarzOptions::new()
            .with_max_plan_bytes(1 << 20)
            .with_max_workspace_bytes(1 << 20);
        let rhs: Vec<_> = (0..n)
            .map(|i| scalar((i as f64 - 2.) / 7., (i as f64 + 1.) / 11.))
            .collect();
        for accuracy in [Accuracy::Standard, Accuracy::Exact] {
            let full_operator = matrix.view().with_accuracy(accuracy);
            let sparse = SchwarzPlan::with_options(
                n,
                SchwarzKind::Multiplicative,
                &domains,
                Some(SchwarzOperatorView::Sparse(matrix.view())),
                limits,
            )
            .unwrap();
            let full = SchwarzPlan::with_options(
                n,
                SchwarzKind::Multiplicative,
                &domains,
                Some(SchwarzOperatorView::MatrixFree(&full_operator)),
                limits,
            )
            .unwrap();
            assert!(
                sparse.domains[..n - 1]
                    .iter()
                    .all(|d| d.affected.len() <= 3)
            );
            assert!(sparse.domains[n - 1].affected.is_empty());
            let mut a = alloc::vec![S::ZERO;n];
            let mut b = a.clone();
            let options = crate::SparseSolveOptions::new().with_accuracy(accuracy);
            let context = ExecutionContext::sequential();
            sparse
                .apply_into(
                    &rhs,
                    options,
                    &mut a,
                    &mut sparse.workspace(1).unwrap(),
                    context,
                )
                .unwrap();
            full.apply_into(
                &rhs,
                options,
                &mut b,
                &mut full.workspace(1).unwrap(),
                context,
            )
            .unwrap();
            for (&a, &b) in a.iter().zip(&b) {
                assert_eq!(
                    a.real_component().decompose(),
                    b.real_component().decompose()
                );
                assert_eq!(
                    a.imaginary_component().decompose(),
                    b.imaginary_component().decompose()
                );
            }
            let peak = sparse.preparation_peak_bytes();
            assert!(
                SchwarzPlan::with_options(
                    n,
                    SchwarzKind::Multiplicative,
                    &domains,
                    Some(SchwarzOperatorView::Sparse(matrix.view())),
                    limits.with_max_plan_bytes(peak - 1)
                )
                .is_err()
            );
        }
    }
    #[test]
    fn sparse_and_full_residual_paths_agree_in_every_domain() {
        compare::<f32>();
        compare::<f64>();
        compare::<scioxide_core::Complex<f32>>();
        compare::<scioxide_core::Complex<f64>>();
    }
}
