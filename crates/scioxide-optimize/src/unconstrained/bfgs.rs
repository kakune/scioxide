//! BFGS with a dense inverse-Hessian approximation.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{
    AttemptKind, AttemptRecord, Capacity, ExecutionContext, NumericScalar, RealScalar,
    SciOxideError, SolveCheckpointMetadata, SolveScope,
};
use scioxide_linalg::Matrix;

use super::settings::{LineSearchSettings, line_search_settings};
use crate::attempt::{Entry, Shell};
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::{LineSearch, ObjectiveFunction};
use crate::unconstrained::FirstOrderStorage;
use crate::utils::{
    finish_first_order, get_gradient_into, line_search_termination, norm2, validate_x0,
};
use crate::{MinimizerAccepted, MinimizerProgress};

/// Sets the stopping tolerance, limits, derivative source, line search,
/// curvature policy and workspace size limit of [`Bfgs`].
///
/// The defaults are the default
/// [`GradientTolerance`](scioxide_core::GradientTolerance), 1000 iterations, the
/// [`DerivativeStrategy::AnalyticOrCentral`](scioxide_differentiate::DerivativeStrategy::AnalyticOrCentral), no
/// time limit, [`StrongWolfeLineSearch`](crate::StrongWolfeLineSearch), the
/// default [`CurvaturePolicy`](crate::CurvaturePolicy) and no workspace size
/// limit. The stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct BfgsOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    settings: LineSearchSettings<S, L>,
    curvature: crate::CurvaturePolicy<S>,
    workspace_byte_limit: usize,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for BfgsOptions<S> {
    fn default() -> Self {
        Self {
            settings: LineSearchSettings::new(crate::line_search::StrongWolfeLineSearch::default()),
            curvature: crate::CurvaturePolicy::default(),
            workspace_byte_limit: usize::MAX,
        }
    }
}

line_search_settings!(BfgsOptions);

impl<S: RealScalar + NumericScalar<Real = S>, L> BfgsOptions<S, L> {
    /// Sets the line search.
    #[must_use]
    pub fn with_line_search<L2>(self, line_search: L2) -> BfgsOptions<S, L2> {
        BfgsOptions {
            settings: self.settings.with_line_search(line_search),
            curvature: self.curvature,
            workspace_byte_limit: self.workspace_byte_limit,
        }
    }

    /// Sets the curvature policy of the updates.
    #[must_use]
    pub fn with_curvature_policy(mut self, policy: crate::CurvaturePolicy<S>) -> Self {
        self.curvature = policy;
        self
    }

    /// Returns the curvature policy of the updates.
    pub const fn curvature_policy(&self) -> crate::CurvaturePolicy<S> {
        self.curvature
    }

    /// Sets the largest workspace size, in bytes.
    #[must_use]
    pub fn with_workspace_byte_limit(mut self, limit: usize) -> Self {
        self.workspace_byte_limit = limit;
        self
    }

    /// Returns the largest workspace size, in bytes.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.workspace_byte_limit
    }
}

/// Describes the workspace that a [`Bfgs`] solve needs, and its size limit.
///
/// The workspace holds about `10 n² + 40 n` values for `n` variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BfgsWorkspaceRequirements {
    dimension: usize,
    bytes: usize,
    limit: usize,
}

impl BfgsWorkspaceRequirements {
    pub(crate) fn new<S: RealScalar>(
        dimension: usize,
        limit: usize,
    ) -> Result<Self, SciOxideError> {
        let square = checked_mul(
            dimension,
            dimension,
            scioxide_core::quantity!("bfgs_matrix_elements"),
        )?;
        // The factor covers the inverse Hessian, transactional update/factor
        // buffers (including Cholesky's retained input and original-input scratch)
        // and matrix-kernel scratch. The linear factor covers iterate, derivative,
        // line-search and staged-result buffers.
        let matrix = checked_mul(square, 10, scioxide_core::quantity!("bfgs_matrix_storage"))?;
        let vectors = checked_mul(
            dimension,
            40,
            scioxide_core::quantity!("bfgs_vector_storage"),
        )?;
        let scalar_elements = checked_add(
            matrix,
            vectors,
            scioxide_core::quantity!("bfgs_workspace_elements"),
        )?;
        let scalar_bytes = checked_mul(
            scalar_elements,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        // The matrix allowance already includes the product candidate. Ask
        // linalg for its current arena bound instead of duplicating panel sizes.
        let product = scioxide_linalg::LinalgWorkspace::<S>::workspace_requirements(
            &scioxide_linalg::DenseWorkspaceRequest::product(
                dimension,
                dimension,
                dimension,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            ),
        )?;
        let candidate_bytes = checked_mul(
            square,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("bfgs_product_candidate_bytes"),
        )?;
        let packing_bytes = product
            .required_bytes::<S>()?
            .checked_sub(candidate_bytes)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("bfgs_product_storage"))
            })?;
        let bytes = checked_add(
            scalar_bytes,
            2048 + packing_bytes,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if bytes > limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                limit,
            ));
        }
        Ok(Self {
            dimension,
            bytes,
            limit,
        })
    }

    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Returns the workspace size in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }

    /// Returns the workspace size limit in bytes.
    pub const fn byte_limit(self) -> usize {
        self.limit
    }
}

/// Stores the state of a [`Bfgs`] solve so that [`Bfgs::resume_into`] can continue it.
///
/// It holds the last accepted point, its gradient and objective value, the
/// inverse-Hessian approximation and the iteration count. Create one with
/// [`BfgsWorkspace::checkpoint`].
#[derive(Debug)]
pub struct BfgsCheckpoint<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) point: Vec<S>,
    pub(crate) gradient: Vec<S>,
    pub(crate) inverse_hessian: Matrix<S>,
    pub(crate) value: S,
    pub(crate) iterations: usize,
    pub(crate) globalization: crate::GlobalizationDiagnostics<S>,
    pub(crate) metadata: SolveCheckpointMetadata,
}

impl<S: RealScalar + NumericScalar<Real = S>> BfgsCheckpoint<S> {
    /// Returns the accepted point.
    pub fn point(&self) -> &[S] {
        &self.point
    }
    /// Returns the gradient at the accepted point.
    pub fn gradient(&self) -> &[S] {
        &self.gradient
    }
    /// Returns the objective value at the accepted point.
    pub const fn value(&self) -> S {
        self.value
    }
    /// Returns the number of completed iterations.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the solve metadata that [`Bfgs::resume_into`] checks.
    pub const fn metadata(&self) -> SolveCheckpointMetadata {
        self.metadata
    }
    pub(crate) fn state_token(&self) -> u128 {
        snapshot_token(
            &self.point,
            &self.gradient,
            &self.inverse_hessian,
            Some(self.value),
            self.iterations,
            self.globalization,
        )
    }
}

fn state_token<S: RealScalar>(storage: &FirstOrderStorage<S>) -> u128 {
    snapshot_token(
        &storage.x,
        &storage.gradient,
        &storage.inverse_hessian,
        storage.accepted_value,
        storage.accepted_iterations,
        storage.globalization,
    )
}

// Checkpoint integrity must cover every numerical lane without formatting a
// dense matrix as decimal text. The exact widening preserves f32/f64 bits,
// including signed zeros; dimensions and scalar metadata delimit the slices.
fn snapshot_token<S: RealScalar>(
    point: &[S],
    gradient: &[S],
    matrix: &Matrix<S>,
    value: Option<S>,
    iterations: usize,
    globalization: crate::GlobalizationDiagnostics<S>,
) -> u128 {
    let initial = crate::context::options_token(&(
        point.len(),
        gradient.len(),
        matrix.rows(),
        matrix.cols(),
        value,
        iterations,
        globalization,
    ));
    let token = point.iter().chain(gradient).fold(initial, |token, &value| {
        (token ^ u128::from(value.into_f64().to_bits())).wrapping_mul(0x100000001b3)
    });
    matrix
        .as_slice()
        .chunks_exact(matrix.cols().max(1))
        .enumerate()
        .fold(token, |token, (row, values)| {
            values[..=row].iter().fold(token, |token, &value| {
                (token ^ u128::from(value.into_f64().to_bits())).wrapping_mul(0x100000001b3)
            })
        })
}

/// Holds the storage and the record of the last call for [`Bfgs`].
///
/// A workspace keeps its storage between solves of the same dimension.
#[derive(Debug)]
pub struct BfgsWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    storage: FirstOrderStorage<S>,
    staged: OptimizeResult<S>,
    shell: Shell<MinimizerProgress<S>>,
    checkpoint_metadata: Option<SolveCheckpointMetadata>,
    retained_bytes: usize,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for BfgsWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> BfgsWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            storage: FirstOrderStorage::new(),
            staged: OptimizeResult::empty(),
            shell: Shell::new(),
            checkpoint_metadata: None,
            retained_bytes: 0,
        }
    }

    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` if the required or current size exceeds the
    /// byte limit of `requirements`.
    pub fn check(
        &self,
        requirements: &BfgsWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        let retained = self.retained_bytes.max(requirements.bytes);
        if retained > requirements.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                retained,
                requirements.limit,
            ));
        }
        Ok(
            if self.storage.fits(requirements.dimension, 0, true)
                && self.staged.fits(requirements.dimension)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Reserves storage for `requirements`.
    ///
    /// On error the existing storage is unchanged.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` if the size exceeds the byte limit, and
    /// `SizeOverflow` or `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &BfgsWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let mut storage = FirstOrderStorage::new();
        storage.try_reserve_bfgs(requirements.dimension)?;
        let mut staged = OptimizeResult::empty();
        staged.try_reserve(requirements.dimension)?;
        self.storage = storage;
        self.staged = staged;
        self.retained_bytes = requirements.bytes;
        Ok(())
    }

    /// Returns the record of the last call to `minimize_into` or `resume_into`.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MinimizerProgress<S>>> {
        self.shell.last_attempt()
    }

    /// Returns the last accepted point of the last call, if it accepted one.
    pub fn last_accepted(&self) -> Option<MinimizerAccepted<'_, S>> {
        crate::attempt::last_accepted(&self.shell, &self.storage)
    }

    /// Copies the state of the last call into a [`BfgsCheckpoint`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the last call accepted no point.
    /// - `AllocationFailure` if the checkpoint cannot be allocated.
    pub fn checkpoint(&self) -> Result<BfgsCheckpoint<S>, SciOxideError> {
        let missing = || {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Malformed,
                "no accepted state in the current attempt",
            )
        };
        let metadata = self.checkpoint_metadata.ok_or_else(missing)?;
        let accepted = self.last_accepted().ok_or_else(missing)?;
        let mut point = Vec::new();
        try_reserve(
            &mut point,
            accepted.point().len(),
            scioxide_core::quantity!("bfgs_checkpoint_point"),
        )?;
        point.extend_from_slice(accepted.point());
        let mut gradient = Vec::new();
        try_reserve(
            &mut gradient,
            self.storage.gradient.len(),
            scioxide_core::quantity!("bfgs_checkpoint_gradient"),
        )?;
        gradient.extend_from_slice(&self.storage.gradient);
        let mut inverse_hessian = Matrix::empty();
        inverse_hessian.try_resize_flat(self.storage.dimension, self.storage.dimension)?;
        inverse_hessian
            .as_mut_slice()
            .copy_from_slice(self.storage.inverse_hessian.as_slice());
        let n = self.storage.dimension;
        for row in 0..n {
            for col in 0..row {
                inverse_hessian.as_mut_slice()[col * n + row] =
                    inverse_hessian.as_slice()[row * n + col];
            }
        }
        Ok(BfgsCheckpoint {
            point,
            gradient,
            inverse_hessian,
            value: accepted.progress().value(),
            iterations: accepted.progress().iterations(),
            globalization: self.storage.globalization,
            metadata,
        })
    }
}

/// Minimizes a smooth function by the BFGS quasi-Newton method with a dense inverse-Hessian approximation.
///
/// The method keeps an `n × n` approximation `H` of the inverse Hessian,
/// starting from the identity, and searches along `−H g`. After each accepted
/// step it applies the BFGS update with the curvature policy of
/// [`BfgsOptions`] (default: Powell's damping). If `−H g` is not a descent
/// direction, `H` is reset to the identity. An iteration costs O(n²)
/// arithmetic plus the line search, and the workspace holds O(n²) values; for
/// many variables use [`Lbfgs`](crate::Lbfgs). The default line search is
/// [`StrongWolfeLineSearch`](crate::StrongWolfeLineSearch).
///
/// [`BfgsWorkspace::checkpoint`] copies the state after a solve, including
/// `H`, and [`Self::resume_into`] continues from it.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{Bfgs, ObjectiveFunction, OptimizeResult};
///
/// struct Sphere;
///
/// impl ObjectiveFunction<f64> for Sphere {
///     fn dim(&self) -> usize {
///         1
///     }
///
///     fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
///         Ok(x[0] * x[0])
///     }
///
///     fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
///         out[0] = 2.0 * x[0];
///         Ok(true)
///     }
/// }
///
/// # fn main() -> Result<(), SciOxideError> {
/// let solver = Bfgs::<f64>::default();
/// let mut workspace = solver.workspace(1)?;
/// let mut output = OptimizeResult::<f64>::empty();
/// solver.minimize_into(
///     &Sphere,
///     &[2.0],
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(output.is_success());
/// // x² has its minimum at x = 0.
/// assert!(output.x()[0].abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §6.1 and §18.3.
#[derive(Debug, Clone)]
pub struct Bfgs<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    options: BfgsOptions<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for Bfgs<S> {
    fn default() -> Self {
        Self::new(BfgsOptions::default())
    }
}

fn reset_inverse_hessian<S>(inverse_hessian: &mut Matrix<S>, dimension: usize)
where
    S: RealScalar + NumericScalar<Real = S>,
{
    inverse_hessian
        .as_mut_slice()
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    (0..dimension).for_each(|index| {
        inverse_hessian.as_mut_slice()[index * dimension + index] =
            <S as scioxide_core::AlgebraicScalar>::ONE;
    });
}

impl<S, L> Bfgs<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: BfgsOptions<S, L>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &BfgsOptions<S, L> {
        &self.options
    }

    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the size overflows `usize`, and `ResourceLimit`
    /// if it exceeds [`BfgsOptions::workspace_byte_limit`].
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<BfgsWorkspaceRequirements, SciOxideError> {
        BfgsWorkspaceRequirements::new::<S>(dimension, self.options.workspace_byte_limit())
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`
    /// if the storage cannot be reserved.
    pub fn workspace(&self, dimension: usize) -> Result<BfgsWorkspace<S>, SciOxideError> {
        let requirements = self.workspace_requirements(dimension)?;
        let mut workspace = BfgsWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }

    pub(crate) fn solve_identity(
        &self,
        problem: scioxide_core::ProblemToken,
        dimension: usize,
    ) -> Result<scioxide_core::SolveIdentity, SciOxideError> {
        crate::context::identity(
            scioxide_core::algorithm_id!("optimize", "BFGS", 1),
            problem,
            dimension,
            crate::context::options_token(&(core::any::type_name::<L>(), &self.options)),
        )
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`BfgsWorkspace::last_attempt`] describes the
    /// failed call.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `f` has bounds.
    /// - `DimensionMismatch` if `x0.len()` differs from `f.dim()`.
    /// - `NonFinite` if `x0`, or the objective or gradient at `x0`, is NaN
    ///   or infinite.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation
    ///   or deadline stops the solve before it accepts a point.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'cap, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut BfgsWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        workspace.checkpoint_metadata = None;
        let scope = scope.into();
        crate::validation::unbounded(f, x0)?;
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        crate::attempt::preflight(&scope, self.options.time_limit())?;
        let entry = Entry::Call {
            identity: self.solve_identity(scioxide_core::ProblemToken::new(0), x0.len())?,
            budgets: self.child_budgets(),
        };
        let checkpoint_metadata = &mut workspace.checkpoint_metadata;
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                self.run_epoch(f, x0, None, context, storage, staged, checkpoint_metadata)
            },
        )
    }

    /// Continues a solve from `checkpoint` and writes the result to `output`.
    ///
    /// The solve continues from the point, inverse-Hessian approximation and
    /// iteration count of the checkpoint. The iteration limit of the options
    /// counts from the start of the original solve, so the resumed run takes at
    /// most `max_iterations − checkpoint iterations` further iterations, and the
    /// reported iteration count includes the checkpoint's. `checkpoint` is not
    /// modified. The checkpoint is checked before `f` is called; on error,
    /// `output` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the parts of `checkpoint` differ in size.
    /// - `InvalidInput` if `f` has bounds, or the checkpoint does not match
    ///   this solver and problem.
    /// - The other errors of [`Self::minimize_into`].
    pub fn resume_into<'scope, 'cap, F>(
        &self,
        f: &F,
        checkpoint: &BfgsCheckpoint<S>,
        output: &mut OptimizeResult<S>,
        workspace: &mut BfgsWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Resume);
        workspace.checkpoint_metadata = None;
        let scope = scope.into();
        let dimension = checkpoint.point.len();
        crate::validation::unbounded(f, &checkpoint.point)?;
        if checkpoint.gradient.len() != dimension
            || checkpoint.inverse_hessian.rows() != dimension
            || checkpoint.inverse_hessian.cols() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(dimension),
                scioxide_core::Dimensions::one(checkpoint.gradient.len()),
            ));
        }
        let identity = self.solve_identity(checkpoint.metadata.identity().problem(), dimension)?;
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        crate::attempt::preflight(&scope, self.options.time_limit())?;
        let entry = Entry::Resume {
            identity,
            state_token: checkpoint.state_token(),
            metadata: checkpoint.metadata,
            budgets: self.child_budgets(),
        };
        let checkpoint_metadata = &mut workspace.checkpoint_metadata;
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                storage.reset();
                storage.x.copy_from_slice(&checkpoint.point);
                storage.gradient.copy_from_slice(&checkpoint.gradient);
                storage
                    .inverse_hessian
                    .as_mut_slice()
                    .copy_from_slice(checkpoint.inverse_hessian.as_slice());
                storage.accepted_value = Some(checkpoint.value);
                storage.accepted_gradient = true;
                storage.accepted_iterations = checkpoint.iterations;
                storage.globalization = checkpoint.globalization;
                self.run_epoch(
                    f,
                    &checkpoint.point,
                    Some((checkpoint.value, checkpoint.iterations)),
                    context,
                    storage,
                    staged,
                    checkpoint_metadata,
                )
            },
        )
    }

    /// Run one fresh or resumed epoch and record checkpoint metadata.
    #[allow(clippy::too_many_arguments)]
    fn run_epoch<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        resume: Option<(S, usize)>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
        storage: &mut FirstOrderStorage<S>,
        staged: &mut OptimizeResult<S>,
        checkpoint_metadata: &mut Option<SolveCheckpointMetadata>,
    ) -> Result<(), SciOxideError> {
        if let Some(limit) = self.options.time_limit() {
            context.limit_time(limit)?;
        }
        let session = crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
        let observed = session.observe(f);
        let outcome = self.minimize_kernel(&observed, x0, staged, storage, &session, None, resume);
        let completed = crate::utils::complete_first_order(outcome, storage, staged, &session);
        if storage.accepted_value.is_some() && storage.accepted_gradient {
            let metadata = session.checkpoint(state_token(storage));
            *checkpoint_metadata = if resume.is_some() {
                metadata.ok()
            } else {
                Some(metadata?)
            };
        }
        completed
    }

    pub(crate) fn child_budgets(&self) -> scioxide_core::SolveBudgets {
        scioxide_core::SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Iteration,
            self.options.max_iterations(),
        )
    }

    pub(crate) fn derivative_strategy(&self) -> scioxide_differentiate::DerivativeStrategy {
        self.options.derivative_strategy()
    }

    pub(crate) fn gradient_threshold(&self) -> Result<S, SciOxideError> {
        self.options
            .gradient_tolerance()
            .threshold(<S as scioxide_core::AlgebraicScalar>::ONE)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn minimize_subproblem<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut FirstOrderStorage<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
        threshold: S,
    ) -> Result<(), SciOxideError> {
        if let Some(limit) = self.options.time_limit() {
            session.limit_time(limit)?;
        }
        let outcome =
            self.minimize_kernel(f, x0, output, workspace, session, Some(threshold), None);
        crate::utils::complete_first_order(outcome, workspace, output, session)
    }

    #[allow(clippy::too_many_arguments)]
    fn minimize_kernel<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut FirstOrderStorage<S>,
        deadline: &crate::context::SolveSession<'_, '_, '_>,
        threshold: Option<S>,
        resume: Option<(S, usize)>,
    ) -> Result<(), SciOxideError> {
        validate_x0(f, x0)?;
        workspace.try_reserve_bfgs(x0.len())?;
        output.try_reserve(x0.len())?;
        if resume.is_none() {
            workspace.reset();
            workspace.x.copy_from_slice(x0);
            reset_inverse_hessian(&mut workspace.inverse_hessian, workspace.dimension);
        }
        let derivative_strategy = self.options.derivative_strategy();
        // Below this conservative L1 bound the Euclidean norm is representable.
        // A component already above the target then certifies non-convergence
        // without constructing the exact norm. Extreme inputs retain its checks.
        let gradient_norm_limit =
            <S as RealScalar>::MAX / S::try_from_usize_exact(workspace.dimension.max(1))?;

        let finite_difference = crate::derivatives::prepared(&mut workspace.finite_difference)?;
        let (mut value, start_iteration) = if let Some((value, iteration)) = resume {
            (value, iteration)
        } else {
            let value = f.eval(&workspace.x)?;
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Callback,
                    "objective returned a non-finite value",
                ));
            }
            workspace.accepted_value = Some(value);
            get_gradient_into(
                f,
                &workspace.x,
                value,
                derivative_strategy,
                &mut workspace.gradient,
                finite_difference,
                deadline.execution()?,
            )?;
            workspace.accepted_gradient = true;
            (value, 0)
        };

        for iteration in start_iteration..self.options.max_iterations().get() {
            if let Some(termination) = deadline.iteration()? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "execution limit reached",
                );
            }
            let target = threshold.unwrap_or(
                self.options
                    .gradient_tolerance()
                    .threshold(crate::utils::gradient_scale(value, &workspace.x))?,
            );
            let maximum = workspace
                .gradient
                .iter()
                .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |m, &g| {
                    m.max(NumericScalar::abs(g))
                });
            let norm_above_target = maximum > target && maximum < gradient_norm_limit;
            if !norm_above_target && norm2(&workspace.gradient)? <= target {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    OptimizeTermination::GradientSmall,
                    "gradient norm below tolerance",
                );
            }
            crate::quasi_newton::trusted_product(
                &workspace.inverse_hessian,
                &workspace.gradient,
                &mut workspace.hessian_product,
                &mut workspace.gemv_scratch,
                deadline,
            )?;
            workspace
                .direction
                .iter_mut()
                .zip(&workspace.hessian_product)
                .for_each(|(direction, &product)| *direction = -product);
            if !crate::quasi_newton::scaled_dot(&workspace.gradient, &workspace.direction)?
                .is_sign_negative()
            {
                reset_inverse_hessian(&mut workspace.inverse_hessian, workspace.dimension);
                workspace
                    .direction
                    .iter_mut()
                    .zip(&workspace.gradient)
                    .for_each(|(direction, &gradient)| *direction = -gradient);
            }
            let directional_derivative = crate::quasi_newton::descent_slope(
                &workspace.gradient,
                &mut workspace.direction,
                workspace.globalization.updates + workspace.globalization.damped_updates == 0,
            )?;
            let trial = self.options.line_search().search_into(
                f,
                &workspace.x,
                value,
                &workspace.direction,
                directional_derivative,
                derivative_strategy,
                &mut workspace.trial_x,
                &mut workspace.trial_gradient,
                &mut workspace.s,
                &mut workspace.y,
                finite_difference,
                crate::LineSearchScope::observed(deadline),
                None,
            )?;
            if let Some(termination) = line_search_termination(trial) {
                let termination = deadline.termination()?.unwrap_or(termination);
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "line search terminated without an accepted trial",
                );
            }
            if let Some(termination) = deadline.termination()? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "execution limit reached during line search",
                );
            }
            if !trial.gradient_available() {
                get_gradient_into(
                    f,
                    &workspace.trial_x,
                    trial.value(),
                    derivative_strategy,
                    &mut workspace.trial_gradient,
                    finite_difference,
                    deadline.execution()?,
                )?;
            } else {
                deadline.cache_hit()?;
            }
            workspace
                .s
                .iter_mut()
                .zip(workspace.trial_x.iter().zip(&workspace.x))
                .for_each(|(step, (&trial, &current))| *step = trial - current);
            workspace
                .y
                .iter_mut()
                .zip(workspace.trial_gradient.iter().zip(&workspace.gradient))
                .for_each(|(difference, (&trial, &current))| *difference = trial - current);
            let update = workspace.update_workspace.prepare(
                workspace.inverse_hessian.view(),
                &workspace.s,
                &workspace.y,
                crate::HessianForm::Inverse,
                crate::HessianUpdate::Bfgs(self.options.curvature_policy()),
                deadline.execution()?,
                Some(deadline),
                if workspace.globalization.updates + workspace.globalization.damped_updates == 0 {
                    crate::quasi_newton::UpdateSource::ScaledIdentity
                } else {
                    crate::quasi_newton::UpdateSource::VerifiedLower
                },
                scioxide_core::argument!("workspace"),
            )?;
            deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
            workspace
                .update_workspace
                .commit(&mut workspace.inverse_hessian, update);
            workspace.globalization.record_update(update);
            core::mem::swap(&mut workspace.x, &mut workspace.trial_x);
            core::mem::swap(&mut workspace.gradient, &mut workspace.trial_gradient);
            value = trial.value();
            workspace.accepted_value = Some(value);
            workspace.accepted_gradient = true;
            workspace.accepted_iterations = iteration + 1;
        }
        finish_first_order(
            workspace,
            output,
            value,
            true,
            self.options.max_iterations().get(),
            OptimizeTermination::IterationLimit,
            "maximum iterations reached",
        )
    }
}

impl<S, L> Bfgs<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(x0.len())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            f,
            x0,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S, L, F> crate::traits::Optimizer<S, F> for Bfgs<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    fn minimize(&self, f: &F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(f, x0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{CancellationToken, CapabilityError, Clock, MonotonicInstant};

    struct Rosenbrock;

    impl ObjectiveFunction<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            Ok(a * a + 100.0 * b * b)
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            output[1] = 200.0 * (x[1] - x[0] * x[0]);
            Ok(true)
        }
    }

    #[test]
    fn lower_model_checkpoint_materializes_and_resumes_without_factorization() {
        struct Interrupted(core::cell::Cell<usize>);
        impl ObjectiveFunction<f64> for Interrupted {
            fn dim(&self) -> usize {
                2
            }
            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                let count = self.0.get() + 1;
                self.0.set(count);
                if count == 20 {
                    Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("x"),
                        scioxide_core::InvalidInputReason::Malformed,
                        "injected failure",
                    ))
                } else {
                    Rosenbrock.eval(x)
                }
            }
            fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
                Rosenbrock.gradient_into(x, out)
            }
        }
        let solver = Bfgs::<f64>::default();
        let mut workspace = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        assert!(
            solver
                .minimize_into(
                    &Interrupted(core::cell::Cell::new(0)),
                    &[-1.2, 1.],
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        let checkpoint = workspace.checkpoint().unwrap();
        assert!(checkpoint.iterations() > 0);
        let matrix = checkpoint.inverse_hessian.as_slice();
        assert_eq!(matrix[1], matrix[2]);
        assert_ne!(matrix[1], 0.);
        assert_eq!(checkpoint.state_token(), state_token(&workspace.storage));
        let mut resumed = solver.workspace(2).unwrap();
        solver
            .resume_into(
                &Rosenbrock,
                &checkpoint,
                &mut output,
                &mut resumed,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success());
        assert_eq!(
            resumed
                .last_attempt()
                .unwrap()
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Factorization),
            0
        );
    }

    #[test]
    fn reusable_bfgs_converges_on_rosenbrock() {
        let solver = Bfgs::<f64>::default();
        let mut workspace = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::<f64>::empty();
        output.try_reserve(2).unwrap();
        solver
            .minimize_into(
                &Rosenbrock,
                &[-1.2, 1.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success());
        assert!(output.fun < 1e-10);
    }

    #[test]
    fn zero_time_limit_before_observation_has_no_partial_result() {
        let solver =
            Bfgs::new(BfgsOptions::<f64>::default().with_time_limit(core::time::Duration::ZERO));
        struct FixedClock;
        impl Clock for FixedClock {
            fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
                Ok(MonotonicInstant::from_duration_since_epoch(
                    core::time::Duration::ZERO,
                ))
            }
        }
        let clock = FixedClock;
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let context = scioxide_core::ExecutionContext::sequential()
            .with_clock(&clock_binding)
            .unwrap();
        let mut workspace = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::<f64>::empty();
        let result = solver.minimize_into(
            &Rosenbrock,
            &[-1.2, 1.0],
            &mut output,
            &mut workspace,
            context,
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::ExecutionStopped {
                reason: scioxide_core::ExecutionStopReason::DeadlineReached,
                ..
            })
        ));
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .total_providers()
                .unwrap(),
            0
        );
    }

    #[test]
    fn cancellation_before_observation_does_not_call_the_objective() {
        let solver = Bfgs::<f64>::default();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let context =
            scioxide_core::ExecutionContext::sequential().with_cancellation(&cancellation);
        let mut workspace = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::<f64>::empty();
        let result = solver.minimize_into(
            &Rosenbrock,
            &[-1.2, 1.0],
            &mut output,
            &mut workspace,
            context,
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::ExecutionStopped {
                reason: scioxide_core::ExecutionStopReason::Cancelled,
                ..
            })
        ));
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .total_providers()
                .unwrap(),
            0
        );
    }
}
