//! Borrowed nonlinear PDE composition; iteration remains in rootfind and ODE.
use crate::validation::finite_slice;
use crate::validation::invalid;
use crate::validation::poison;
use crate::validation::shape;

use alloc::vec::Vec;
use core::{cell::RefCell, ops::Range};
use scioxide_core::backend::{checked_add, try_filled, try_with_capacity};
use scioxide_core::{
    AdjointOperator, ConvergenceDiagnostics, ConvergenceReason, EvaluationRole, InvalidInputReason,
    JacobianOperator, LinearOperator, NonFiniteOrigin, NumericScalar, Preconditioner, RealScalar,
    ResidualOperator, SciOxideError, SolveContext, StructuralPattern,
};
use scioxide_linalg::backend::dense::DenseOperatorWorkspace;
use scioxide_ode::OdeSystem;
use scioxide_sparse::CsrMatrix;

/// The plan's provider (held by the entry point's `self`) lacks the selected derivative.
fn unsupported() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("self"),
        InvalidInputReason::UnsupportedConfiguration,
        "the explicitly selected derivative capability is unavailable",
    )
}

fn assembled_product<T: NumericScalar>(
    matrix: &dyn AdjointOperator<T>,
    direction: &[T],
    output: &mut [T],
    scratch: &mut [T],
    adjoint: bool,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let mut operator = scioxide_core::OperatorContext::new(context);
    if adjoint {
        matrix.apply_adjoint_into_scoped(direction, output, scratch, &mut operator)
    } else {
        matrix.apply_into_scoped(direction, output, scratch, &mut operator)
    }
    .and_then(scioxide_core::Invocation::require)
}
/// Run an unscoped evaluation in a private sequential epoch whose counts are
/// discarded; scoped evaluations charge the caller's context instead.
fn unscoped<R>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError> {
    crate::attempt::sequential(
        scioxide_core::algorithm_id!("pde", "nonlinear-residual", 2),
        run,
    )
}

/// Checked contiguous field blocks in a coupled system's free-coordinate order.
///
/// Blocks may have unequal or zero lengths (fully constrained fields). The total
/// dimension must be positive. Mesh identity and reconstruction remain with the
/// discretization's typed fields. Metadata takes O(number of fields) storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PdeFieldLayout {
    offsets: Vec<usize>,
}
impl PdeFieldLayout {
    /// Copy field lengths into checked prefix offsets.
    /// # Errors
    /// Rejects an empty total state, size overflow or allocation failure.
    pub fn new(lengths: &[usize]) -> Result<Self, SciOxideError> {
        let mut offsets = try_with_capacity(
            checked_add(lengths.len(), 1, scioxide_core::quantity!("offsets"))?,
            scioxide_core::quantity!("offsets"),
        )?;
        offsets.push(0);
        let mut total = 0;
        for &length in lengths {
            total = checked_add(total, length, scioxide_core::quantity!("dimension"))?;
            offsets.push(total);
        }
        if total == 0 {
            return Err(invalid(
                scioxide_core::argument!("lengths"),
                "at least one free coordinate is required",
            ));
        }
        Ok(Self { offsets })
    }
    /// Number of component fields.
    pub fn fields(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Total number of free coordinates.
    pub fn dimension(&self) -> usize {
        self.offsets[self.fields()]
    }
    /// Contiguous coordinate range of a field.
    /// # Errors
    /// Rejects an invalid field index.
    pub fn range(&self, field: usize) -> Result<Range<usize>, SciOxideError> {
        if field >= self.fields() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("fields"),
                field,
                self.fields(),
            ));
        }
        Ok(self.offsets[field]..self.offsets[field + 1])
    }

    /// Borrow a field from a full coupled state or residual.
    /// # Errors
    /// Rejects a mismatched state length or invalid field index.
    pub fn field<'a, T>(&self, values: &'a [T], field: usize) -> Result<&'a [T], SciOxideError> {
        shape(values.len(), self.dimension())?;
        Ok(&values[self.range(field)?])
    }
    /// Exclusively borrow a field from caller-owned coupled storage.
    /// # Errors
    /// Rejects a mismatched state length or invalid field index.
    pub fn field_mut<'a, T>(
        &self,
        values: &'a mut [T],
        field: usize,
    ) -> Result<&'a mut [T], SciOxideError> {
        shape(values.len(), self.dimension())?;
        Ok(&mut values[self.range(field)?])
    }
}

/// Block-diagonal inverse action borrowing one prepared preconditioner per field.
///
/// No factorization or nested solve is hidden here. Block providers own their
/// preparation and must keep dimensions and scratch sizes fixed. A composite
/// application counts as one preconditioner evaluation in the invoking solver.
pub struct PdeFieldPreconditioner<'a, T: NumericScalar> {
    fields: &'a PdeFieldLayout,
    blocks: &'a [&'a dyn Preconditioner<T>],
    sizes: Vec<usize>,
    scratch: usize,
}
impl<'a, T: NumericScalar> PdeFieldPreconditioner<'a, T> {
    /// Validate a complete field partition and record exact block scratch sizes.
    /// # Errors
    /// Rejects incompatible blocks, size overflow and allocation failures.
    pub fn new(
        fields: &'a PdeFieldLayout,
        blocks: &'a [&'a dyn Preconditioner<T>],
    ) -> Result<Self, SciOxideError> {
        shape(blocks.len(), fields.fields())?;
        let mut sizes = try_with_capacity(
            blocks.len(),
            scioxide_core::quantity!("block_workspace_sizes"),
        )?;
        let mut maximum = 0;
        for (i, block) in blocks.iter().enumerate() {
            shape(block.dimension(), fields.range(i)?.len())?;
            let size = block.precondition_workspace_len()?;
            maximum = maximum.max(size);
            sizes.push(size);
        }
        Ok(Self {
            fields,
            blocks,
            sizes,
            scratch: checked_add(
                fields.dimension(),
                maximum,
                scioxide_core::quantity!("scratch"),
            )?,
        })
    }
}
impl<T: NumericScalar> Preconditioner<T> for PdeFieldPreconditioner<'_, T> {
    fn dimension(&self) -> usize {
        self.fields.dimension()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn precondition_into(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        shape(input.len(), self.dimension())?;
        shape(output.len(), self.dimension())?;
        if scratch.len() < self.scratch {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                "insufficient preconditioner scratch",
            ));
        }
        finite_slice(input, NonFiniteOrigin::Input)?;
        // Check every provider before invoking the first one.
        for (i, block) in self.blocks.iter().enumerate() {
            shape(block.dimension(), self.fields.range(i)?.len())?;
            shape(block.precondition_workspace_len()?, self.sizes[i])?;
        }
        let (staged, work) = scratch.split_at_mut(self.dimension());
        poison(staged);
        for (i, block) in self.blocks.iter().enumerate() {
            let range = self.fields.range(i)?;
            block.precondition_into(
                &input[range.clone()],
                &mut staged[range],
                &mut work[..self.sizes[i]],
            )?;
        }
        finite_slice(staged, NonFiniteOrigin::Callback)?;
        output.copy_from_slice(staged);
        Ok(())
    }
}

/// Explicit derivative representation; unavailable capabilities never fall back.
#[derive(Clone, Debug)]
pub enum PdeJacobianMode {
    /// Dense analytical Jacobian, with O(n²) reserved matrix storage.
    Dense,
    /// Canonical CSR analytical Jacobian with a fixed complete structural pattern.
    Sparse(StructuralPattern),
    /// Supplied JVP with its maximum scalar scratch requirement; no matrix storage.
    Operator {
        /// Scratch passed to `OdeSystem::jacobian_product`.
        operator_scratch: usize,
    },
}

/// Optional matrix-free conjugate-adjoint product at a physical time and state.
///
/// The callback writes every output component, using only declared scratch.
/// It must describe the same complex-linear derivative as the supplied JVP.
pub type PdeAdjointCallback<'a, T> = dyn Fn(<T as NumericScalar>::Real, &[T], &[T], &mut [T], &mut [T]) -> Result<(), SciOxideError>
    + 'a;

/// A physical provider evaluated at a fixed time, with an optional matrix-free adjoint.
#[derive(Clone, Copy)]
#[must_use]
pub struct NonlinearPdeProblem<'a, T: NumericScalar> {
    fields: &'a PdeFieldLayout,
    mode: &'a PdeJacobianMode,
    provider: &'a dyn OdeSystem<T::Real, T>,
    time: T::Real,
    adjoint: Option<(&'a PdeAdjointCallback<'a, T>, usize)>,
}

impl<'a, T: NumericScalar> NonlinearPdeProblem<'a, T> {
    /// Evaluate `provider` at `time`.
    pub const fn new(
        provider: &'a dyn OdeSystem<T::Real, T>,
        time: T::Real,
        fields: &'a PdeFieldLayout,
        mode: &'a PdeJacobianMode,
    ) -> Self {
        Self {
            fields,
            mode,
            provider,
            time,
            adjoint: None,
        }
    }
    /// Supply the matrix-free adjoint and its scratch; assembled modes use their own
    /// transpose, and binding rejects an adjoint for them.
    pub const fn with_adjoint(
        mut self,
        callback: &'a PdeAdjointCallback<'a, T>,
        scratch: usize,
    ) -> Self {
        self.adjoint = Some((callback, scratch));
        self
    }
    /// The physical provider.
    pub const fn provider(&self) -> &'a dyn OdeSystem<T::Real, T> {
        self.provider
    }
    /// Evaluation time.
    pub const fn time(&self) -> T::Real {
        self.time
    }
}

enum DerivativeStorage<T: NumericScalar> {
    Dense(DenseOperatorWorkspace<T>),
    Sparse(CsrMatrix<T>),
    Operator,
}
/// Caller-owned derivative storage; matrix-free mode reserves no matrix.
///
/// Dense and sparse modes retain one validated Jacobian and O(n) state storage.
/// Products reuse it while the derivative state is unchanged. Binding resets
/// reuse; provider data must remain unchanged during a binding.
pub struct NonlinearPdeWorkspace<T: NumericScalar> {
    values: scioxide_sparse::SparseValueWorkspace<T>,
    storage: DerivativeStorage<T>,
    point: Vec<T>,
    valid: bool,
}
impl<'a, T: NumericScalar> NonlinearPdeProblem<'a, T> {
    /// Check the derivative representation against the field layout.
    /// # Errors
    /// Rejects mismatched patterns and size overflow.
    fn product_scratch(&self) -> Result<usize, SciOxideError> {
        let n = self.fields.dimension();
        let work = match self.mode {
            PdeJacobianMode::Sparse(p) => {
                shape(p.view().rows(), n)?;
                shape(p.view().cols(), n)?;
                n
            }
            PdeJacobianMode::Dense => {
                n.checked_mul(n).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("dense_jacobian_entries"))
                })?;
                n
            }
            PdeJacobianMode::Operator { operator_scratch } => *operator_scratch,
        };
        checked_add(n, work, scioxide_core::quantity!("product_scratch"))
    }
    /// Field block metadata.
    pub const fn fields(&self) -> &'a PdeFieldLayout {
        self.fields
    }
    /// Derivative representation.
    pub const fn mode(&self) -> &'a PdeJacobianMode {
        self.mode
    }
    /// Allocate representation-specific storage once, without invoking numerical callbacks.
    /// # Errors
    /// Returns size, allocation or sparse construction errors.
    pub fn workspace(&self) -> Result<NonlinearPdeWorkspace<T>, SciOxideError> {
        self.product_scratch()?;
        let n = self.fields.dimension();
        let storage = match &self.mode {
            PdeJacobianMode::Dense => DerivativeStorage::Dense(DenseOperatorWorkspace::new(n, n)?),
            PdeJacobianMode::Sparse(p) => DerivativeStorage::Sparse(CsrMatrix::from_pattern(
                p.clone(),
                try_filled(
                    p.view().col_indices().len(),
                    T::ZERO,
                    scioxide_core::quantity!("jacobian_values"),
                )?,
            )?),
            PdeJacobianMode::Operator { .. } => DerivativeStorage::Operator,
        };
        let values = match &storage {
            DerivativeStorage::Sparse(matrix) => matrix.workspace(())?,
            _ => scioxide_sparse::SparseValueWorkspace::new(),
        };
        let point = if matches!(storage, DerivativeStorage::Operator) {
            Vec::new()
        } else {
            try_filled(n, T::ZERO, scioxide_core::quantity!("linearization_point"))?
        };
        Ok(NonlinearPdeWorkspace {
            storage,
            values,
            point,
            valid: false,
        })
    }
    /// Bind the physical provider and an exclusive workspace, resetting Jacobian reuse.
    /// Provider data must remain unchanged until the binding is dropped; rebind
    /// after changing captured model data, even at the same time and state.
    /// # Errors
    /// Rejects a provider of another dimension or with its own mass, a nonfinite time,
    /// an adjoint for assembled modes and incompatible storage, before callbacks.
    pub fn bind(
        self,
        workspace: &'a mut NonlinearPdeWorkspace<T>,
    ) -> Result<PdeResidual<'a, T>, SciOxideError> {
        let n = self.fields.dimension();
        let product_scratch = self.product_scratch()?;
        let problem = self;
        shape(problem.provider.dimension(), n)?;
        if problem.provider.mass_matrix_revision().is_some() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "supply physical mass separately through MethodOfLinesPlan",
            ));
        }
        if !problem.time.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "time must be finite",
            ));
        }
        if let Some((_, scratch)) = problem.adjoint {
            if !matches!(self.mode, PdeJacobianMode::Operator { .. }) {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "assembled derivatives already supply an adjoint",
                ));
            }
            checked_add(n, scratch, scioxide_core::quantity!("adjoint_scratch"))?;
        }
        let compatible = match (&workspace.storage, &self.mode) {
            (DerivativeStorage::Dense(m), PdeJacobianMode::Dense) => m.rows() == n && m.cols() == n,
            (DerivativeStorage::Sparse(m), PdeJacobianMode::Sparse(p)) => {
                let p = p.view();
                m.rows() == p.rows()
                    && m.cols() == p.cols()
                    && m.row_ptr() == p.row_ptr()
                    && m.col_indices() == p.col_indices()
            }
            (DerivativeStorage::Operator, PdeJacobianMode::Operator { .. }) => true,
            _ => false,
        };
        if !compatible {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace does not match the selected derivative representation",
            ));
        }
        workspace.valid = false;
        Ok(PdeResidual {
            product_scratch,
            problem,
            workspace: RefCell::new(workspace),
        })
    }
}

/// Fixed-time core residual/JVP adapter borrowing exclusive derivative storage.
///
/// Pass this directly to `NewtonKrylovProblem::with_jacobian` (or `numerical`).
/// Reentrant workspace use returns a structured error. All outputs are transactional.
/// Scoped applications pass the caller's context to every provider callback; unscoped
/// ones give the callbacks a private, unbounded sequential epoch whose counts are
/// discarded. JVP scratch also covers adjoint products.
pub struct PdeResidual<'a, T: NumericScalar> {
    product_scratch: usize,
    problem: NonlinearPdeProblem<'a, T>,
    workspace: RefCell<&'a mut NonlinearPdeWorkspace<T>>,
}
impl<T: NumericScalar> PdeResidual<'_, T> {
    fn check(
        &self,
        point: &[T],
        output: &[T],
        scratch: &[T],
        required: usize,
    ) -> Result<(), SciOxideError> {
        shape(self.problem.provider.dimension(), self.input_dimension())?;
        if self.problem.provider.mass_matrix_revision().is_some() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "provider mass changed after preparation",
            ));
        }
        shape(point.len(), self.input_dimension())?;
        shape(output.len(), self.output_dimension())?;
        if scratch.len() < required {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                "insufficient nonlinear PDE scratch",
            ));
        }
        finite_slice(point, NonFiniteOrigin::Input)
    }
    fn residual(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.input_dimension();
        self.check(point, output, scratch, n)?;
        context.execution().check()?;
        let staged = &mut scratch[..n];
        poison(staged);
        self.problem
            .provider
            .rhs(self.problem.time, point, staged, context)?;
        finite_slice(staged, NonFiniteOrigin::Callback)?;
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
    fn product(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
        adjoint: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let required = if adjoint {
            self.adjoint_scratch()?
        } else {
            self.product_scratch
        };
        self.check(point, output, scratch, required)?;
        shape(direction.len(), self.input_dimension())?;
        finite_slice(direction, NonFiniteOrigin::Input)?;
        context.execution().check()?;
        let (staged, scratch) = scratch[..required].split_at_mut(self.input_dimension());
        poison(staged);
        let mut workspace = self.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "derivative workspace is already borrowed",
            )
        })?;
        let (provider, time) = (self.problem.provider, self.problem.time);
        self.assemble(point, &mut workspace, context)?;
        match &workspace.storage {
            DerivativeStorage::Dense(m) => {
                assembled_product(&m.operator()?, direction, staged, scratch, adjoint, context)?;
            }
            DerivativeStorage::Sparse(m) => {
                assembled_product(m, direction, staged, scratch, adjoint, context)?;
            }
            DerivativeStorage::Operator => {
                if adjoint {
                    let (callback, _) = self.problem.adjoint.ok_or_else(unsupported)?;
                    callback(time, point, direction, staged, scratch)?;
                } else {
                    provider
                        .jacobian_product(time, point, direction, staged, scratch, context)?
                        .ok_or_else(unsupported)?;
                }
            }
        }
        finite_slice(staged, NonFiniteOrigin::Callback)?;
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
    fn assemble(
        &self,
        point: &[T],
        workspace: &mut NonlinearPdeWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if matches!(workspace.storage, DerivativeStorage::Operator)
            || (workspace.valid && workspace.point == point)
        {
            return Ok(());
        }
        workspace.valid = false;
        if let scioxide_core::Admission::Denied(report) =
            context.record_provider(EvaluationRole::Jacobian)?
        {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                ConvergenceDiagnostics::new()
                    .with_counts(context.counts())
                    .with_budget(report),
                "Jacobian assembly budget exhausted",
            ));
        }
        let (provider, time) = (self.problem.provider, self.problem.time);
        match &mut workspace.storage {
            DerivativeStorage::Dense(matrix) => matrix.update(|candidate| {
                provider
                    .jacobian(time, point, candidate, context)?
                    .ok_or_else(unsupported)
            })?,
            DerivativeStorage::Sparse(matrix) => {
                let PdeJacobianMode::Sparse(pattern) = &self.problem.mode else {
                    return Err(invalid(
                        scioxide_core::argument!("self"),
                        "incompatible sparse storage",
                    ));
                };
                matrix.update_values(
                    |_, candidate| {
                        provider
                            .sparse_jacobian(time, point, pattern.view(), candidate, context)?
                            .ok_or_else(unsupported)
                    },
                    &mut workspace.values,
                )?;
            }
            DerivativeStorage::Operator => return Ok(()),
        }
        context.execution().check()?;
        workspace.point.copy_from_slice(point);
        workspace.valid = true;
        Ok(())
    }
    fn adjoint_scratch(&self) -> Result<usize, SciOxideError> {
        match self.problem.mode {
            PdeJacobianMode::Operator { .. } => checked_add(
                self.input_dimension(),
                self.problem.adjoint.ok_or_else(unsupported)?.1,
                scioxide_core::quantity!("adjoint_scratch"),
            ),
            _ => Ok(self.product_scratch),
        }
    }
}
impl<T: NumericScalar> ResidualOperator<T> for PdeResidual<'_, T> {
    fn input_dimension(&self) -> usize {
        self.problem.fields.dimension()
    }
    fn output_dimension(&self) -> usize {
        self.input_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.input_dimension())
    }
    fn residual_into(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        unscoped(|context| self.residual(point, output, scratch, context))
    }
    fn residual_into_scoped(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.residual(point, output, scratch, context)
    }
}
impl<T: NumericScalar> JacobianOperator<T> for PdeResidual<'_, T> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(match self.adjoint_scratch() {
            Ok(adjoint) => adjoint.max(self.product_scratch),
            Err(_) => self.product_scratch,
        })
    }
    fn jvp_into(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        unscoped(|context| self.product(point, direction, output, scratch, false, context))
    }
    fn jvp_into_scoped(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.product(point, direction, output, scratch, false, context)
    }
}
impl<T: NumericScalar> PdeAdjointJacobian<T> for PdeResidual<'_, T> {
    fn vjp_into(
        &self,
        point: &[T],
        seed: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        unscoped(|context| self.product(point, seed, output, scratch, true, context))
    }
    fn vjp_into_scoped(
        &self,
        point: &[T],
        seed: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.product(point, seed, output, scratch, true, context)
    }
}

/// A residual with conjugate-transpose Jacobian products, so its fixed-state
/// [`PdeTangent`] is an [`AdjointOperator`]. Adjoint products use at most
/// [`JacobianOperator::jvp_workspace_len`] scratch.
pub trait PdeAdjointJacobian<T: NumericScalar>: JacobianOperator<T> {
    /// Evaluate `J(point)ᴴ seed`.
    /// # Errors
    /// Returns shape, domain, provider or numerical errors.
    fn vjp_into(
        &self,
        point: &[T],
        seed: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError>;
    /// Adjoint product with nested work charged to the invoking context; the caller
    /// has already admitted the operator application.
    /// # Errors
    /// Propagates product and nested-work failures.
    fn vjp_into_scoped(
        &self,
        point: &[T],
        seed: &[T],
        output: &mut [T],
        scratch: &mut [T],
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.vjp_into(point, seed, output, scratch)
    }
}

/// Borrowed Jacobian of a residual provider at one fixed, checked point.
///
/// No derivative is evaluated at construction; every application calls a
/// provider JVP (or, for [`PdeAdjointJacobian`] providers, an adjoint product).
/// [`PdeResidual`] reuses assembled derivatives at the same state. One
/// type serves [`PdeResidual`], [`crate::HyperelasticityPlan`] and
/// [`crate::NavierStokesPlan`].
pub struct PdeTangent<'a, T: NumericScalar, P: ?Sized> {
    provider: &'a P,
    point: &'a [T],
}
impl<T: NumericScalar, P: ?Sized> Clone for PdeTangent<'_, T, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: NumericScalar, P: ?Sized> Copy for PdeTangent<'_, T, P> {}
impl<'a, T: NumericScalar, P: JacobianOperator<T> + ?Sized> PdeTangent<'a, T, P> {
    /// Borrow the Jacobian of a square `provider` at `point`.
    /// # Errors
    /// Rejects a non-square provider, a mismatched point and nonfinite coordinates.
    pub fn new(provider: &'a P, point: &'a [T]) -> Result<Self, SciOxideError> {
        shape(provider.output_dimension(), provider.input_dimension())?;
        shape(point.len(), provider.input_dimension())?;
        finite_slice(point, NonFiniteOrigin::Input)?;
        Ok(Self { provider, point })
    }
    /// The linearization point.
    pub const fn point(&self) -> &'a [T] {
        self.point
    }
}
impl<T: NumericScalar, P: JacobianOperator<T> + ?Sized> LinearOperator<T> for PdeTangent<'_, T, P> {
    fn rows(&self) -> usize {
        self.provider.output_dimension()
    }
    fn cols(&self) -> usize {
        self.provider.input_dimension()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.jvp_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.provider.jvp_into(self.point, input, output, scratch)
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.provider
            .jvp_into_scoped(self.point, input, output, scratch, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<T: NumericScalar, P: PdeAdjointJacobian<T> + ?Sized> AdjointOperator<T>
    for PdeTangent<'_, T, P>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.jvp_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.provider.vjp_into(self.point, input, output, scratch)
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.provider
            .vjp_into_scoped(self.point, input, output, scratch, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<T: NumericScalar, P: JacobianOperator<T> + ?Sized> scioxide_core::BlockOperator<T>
    for PdeTangent<'_, T, P>
{
}

#[cfg(test)]
mod tests;
