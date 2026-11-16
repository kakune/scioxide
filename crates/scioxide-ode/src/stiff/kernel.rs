//! The stiff stage kernel: linearizations, mass, effective factors and stage solves.
//!
//! A kernel prepares `E = I_s ⊗ M − H ⊗ J` for `s` simultaneous stage blocks, where
//! `H = h·A` comes from the method's [`Coupling`]. Simplified Newton shares one Jacobian
//! between all blocks; full Newton on coupled stages keeps one Jacobian per stage point.
//! Dense storage assembles `E` and factors it with a pivoted LU, sparse storage assembles
//! it on the blocked pattern and factors it with a sparse LU, and Krylov storage applies
//! it matrix-free. Mass matrices other than the identity require dense storage.
//! Radau's three stage blocks instead share one Jacobian and split into `n×n`
//! eigenmodes through [`radau`]; neither its dense nor sparse backend assembles E.
//!
//! Jacobians are reused within one epoch for the same derivative strategy until relative
//! time/state movement exceeds 0.5 or Newton feedback requests refresh. Constant
//! Jacobians ignore movement and feedback. Coupled simplified Newton may keep a
//! factor while its effective-step ratio stays in [0.8, 1.2]; operator coefficients
//! remain current. Rosenbrock and Radau use exact step-dependent factors.
use super::accuracy::StageAccuracy;
use super::jacobian::{Differences, ProductPoint, SparseColumns};
use super::linear::{LinearStorage, OdeStageBackend, iterative_tolerance, verify_defect};
use super::reuse;
use crate::OdeSystem;
use crate::safety::{finite, invalid};
use crate::stepper::{StepRequest, admit_provider};
use crate::traits::StageLinearSolver;
use alloc::vec::Vec;
use core::cell::Cell;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    EvaluationRole, LinearOperator, NumericScalar, Preconditioner, ProblemToken, RealScalar,
    SciOxideError, SolveBudgets, SolveContext, SolveScope, SolveState, StructuralPattern,
};
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy};
use scioxide_linalg::{Matrix, backend::lu::LuFactor};

mod radau;
use radau::RadauStorage;

/// The stage vectors may be coupled even when their linear systems split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageLayout {
    Coupled,
    Radau,
}

/// Revision of the constant mass matrix used by a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MassKey {
    /// The identity mass matrix.
    Identity,
    /// A provider matrix with this revision.
    Revision(u64),
}

/// Exact cumulative cache activity of a kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CacheStats {
    pub(crate) jacobian_hits: usize,
    pub(crate) factor_hits: usize,
    pub(crate) jacobian_invalidations: usize,
    pub(crate) factor_invalidations: usize,
}

/// Coupling of `s` stage blocks: `H = step · coefficients` (row-major `s × s`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coupling<'a, S> {
    /// Start of the step, passed to coupled preconditioners.
    pub(crate) time: S,
    pub(crate) step: S,
    pub(crate) coefficients: &'a [S],
    pub(crate) nodes: &'a [S],
}

/// Requested storage shape of a kernel; the backend is borrowed so that checking a
/// prepared kernel never copies a sparse pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelShape<'a> {
    pub(crate) backend: &'a OdeStageBackend,
    pub(crate) layout: StageLayout,
    /// Simultaneous stage blocks `s`.
    pub(crate) stages: usize,
    /// Distinct Jacobians: one, or `s` for full Newton on coupled stages.
    pub(crate) blocks: usize,
    /// An injected linear or nonlinear provider replaces factorization.
    pub(crate) provided: bool,
}

// One prepared variant lives inline for the kernel's lifetime.
#[allow(clippy::large_enum_variant)]
enum Jacobians<T: NumericScalar> {
    Dense(Vec<Matrix<T>>),
    Sparse {
        columns: SparseColumns,
        values: Vec<T>,
    },
    Operator {
        sources: Vec<Cell<Option<DerivativeCandidate>>>,
        scratch: usize,
    },
}

/// Read-only linearization applied by the effective operator.
struct Linearization<S: RealScalar, T: NumericScalar<Real = S>> {
    n: usize,
    stages: usize,
    full: bool,
    jacobians: Jacobians<T>,
    points: Vec<T>,
    rhs: Vec<T>,
    times: Vec<S>,
    /// Absolute-tolerance floors of finite-difference steps.
    floors: Vec<S>,
    states: Vec<T>,
    mass: Option<Matrix<T>>,
    strategy: DerivativeStrategy,
    time: S,
    step: S,
    coefficients: Vec<S>,
    nodes: Vec<S>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FactorKey<S> {
    revision: u64,
    mass: MassKey,
    step: S,
    full: bool,
}

struct Cache<S> {
    valid: bool,
    strategy: Option<DerivativeStrategy>,
    time: Option<S>,
    constant: bool,
    revision: u64,
    factor: Option<FactorKey<S>>,
    mass: Option<MassKey>,
    refresh: bool,
    accepted: usize,
}

// Exactly one prepared layout lives inline for the kernel's lifetime.
#[allow(clippy::large_enum_variant)]
enum EffectiveStorage<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    Provided,
    Coupled(LinearStorage<T>),
    Radau(RadauStorage<S, T>),
}

/// Jacobians, mass, effective factor and solver storage of one stiff stage system.
pub(crate) struct StageKernel<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    backend: OdeStageBackend,
    layout: StageLayout,
    blocks: usize,
    provided: bool,
    lin: Linearization<S, T>,
    linear: EffectiveStorage<S, T>,
    differences: Differences<T>,
    mass_factor: LuFactor<T>,
    mass_factor_valid: bool,
    cache: Cache<S>,
    stats: CacheStats,
    factor_coefficients: Vec<S>,
    exact_jacobian: bool,
    lagged_jacobian: bool,
    lagged_factor: bool,
    /// Last measured Newton contraction rate in this epoch.
    contraction: Option<S>,
    product: Vec<T>,
    scratch: Vec<T>,
    krylov: SolveState,
    evaluation: SolveState,
}

fn count(value: &mut usize, quantity: scioxide_core::Quantity) -> Result<(), SciOxideError> {
    *value = scioxide_core::backend::checked_add(*value, 1, quantity)?;
    Ok(())
}

/// The `s·n` pattern whose `s × s` blocks all repeat `pattern`.
fn coupled_pattern(
    pattern: &StructuralPattern,
    s: usize,
) -> Result<StructuralPattern, SciOxideError> {
    let p = pattern.view();
    let n = p.rows();
    let size = checked_mul(n, s, scioxide_core::quantity!("coupled_stage_dimension"))?;
    let entries = checked_mul(
        checked_mul(
            p.col_indices().len(),
            s,
            scioxide_core::quantity!("coupled_entries"),
        )?,
        s,
        scioxide_core::quantity!("coupled_entries"),
    )?;
    let mut row_ptr = try_filled(size + 1, 0usize, scioxide_core::quantity!("coupled_rows"))?;
    let mut columns = try_filled(entries, 0usize, scioxide_core::quantity!("coupled_columns"))?;
    let mut next = 0;
    for block_row in 0..s {
        for row in 0..n {
            for block_column in 0..s {
                for &column in &p.col_indices()[p.row_ptr()[row]..p.row_ptr()[row + 1]] {
                    columns[next] = block_column * n + column;
                    next += 1;
                }
            }
            row_ptr[block_row * n + row + 1] = next;
        }
    }
    StructuralPattern::new(size, size, row_ptr, columns)
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> StageKernel<S, T> {
    /// Reserve every buffer of a stage system with `n` unknowns per block.
    pub(crate) fn new(n: usize, shape: KernelShape<'_>) -> Result<Self, SciOxideError> {
        let s = shape.stages;
        if s == 0
            || !(shape.blocks == 1 || shape.blocks == s)
            || (shape.layout == StageLayout::Radau && (s != 3 || shape.blocks != 1))
        {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "stage blocks must be one or the stage count",
            ));
        }
        shape.backend.validate(n)?;
        let size = checked_mul(n, s, scioxide_core::quantity!("coupled_stage_dimension"))?;
        let points = checked_mul(n, shape.blocks, scioxide_core::quantity!("stage_points"))?;
        let difference_scratch = checked_mul(n, 3, scioxide_core::quantity!("product_scratch"))?;
        let (jacobians, effective) = match shape.backend {
            OdeStageBackend::Dense => {
                let mut blocks = Vec::new();
                scioxide_core::backend::try_reserve(
                    &mut blocks,
                    shape.blocks,
                    scioxide_core::quantity!("stage_jacobians"),
                )?;
                for _ in 0..shape.blocks {
                    let mut matrix = Matrix::empty();
                    matrix.try_resize_flat(n, n)?;
                    blocks.push(matrix);
                }
                (Jacobians::Dense(blocks), OdeStageBackend::Dense)
            }
            OdeStageBackend::Sparse { pattern, limits } => {
                let columns = SparseColumns::new(pattern)?;
                let values = try_filled(
                    checked_mul(
                        columns.len(),
                        shape.blocks,
                        scioxide_core::quantity!("values"),
                    )?,
                    T::ZERO,
                    scioxide_core::quantity!("sparse_jacobian.values"),
                )?;
                let effective = OdeStageBackend::Sparse {
                    pattern: if s == 1 || shape.layout == StageLayout::Radau || shape.provided {
                        pattern.clone()
                    } else {
                        coupled_pattern(pattern, s)?
                    },
                    limits: *limits,
                };
                (Jacobians::Sparse { columns, values }, effective)
            }
            OdeStageBackend::Krylov {
                iterations,
                restart,
                operator_scratch,
                preconditioner_scratch,
            } => {
                let scratch = scioxide_core::backend::checked_add(
                    scioxide_core::backend::checked_add(
                        *operator_scratch,
                        size,
                        scioxide_core::quantity!("operator_scratch"),
                    )?,
                    difference_scratch,
                    scioxide_core::quantity!("operator_scratch"),
                )?;
                let sources = (0..shape.blocks).map(|_| Cell::new(None)).collect();
                (
                    Jacobians::Operator {
                        sources,
                        scratch: *operator_scratch,
                    },
                    OdeStageBackend::Krylov {
                        iterations: *iterations,
                        restart: *restart,
                        operator_scratch: scratch,
                        preconditioner_scratch: *preconditioner_scratch,
                    },
                )
            }
        };
        let operator_scratch = match &effective {
            OdeStageBackend::Krylov {
                operator_scratch, ..
            } => *operator_scratch,
            _ => 0,
        };
        let linear = if shape.layout == StageLayout::Radau {
            EffectiveStorage::Radau(RadauStorage::new(n, shape.backend, shape.provided)?)
        } else if shape.provided {
            EffectiveStorage::Provided
        } else {
            EffectiveStorage::Coupled(LinearStorage::new(size, &effective)?)
        };
        let mass = if matches!(*shape.backend, OdeStageBackend::Dense) {
            let mut matrix = Matrix::empty();
            matrix.try_resize_flat(n, n)?;
            Some(matrix)
        } else {
            None
        };
        let mass_factor = LuFactor::new(if mass.is_some() { n } else { 0 })?;
        let filled = |len, quantity| try_filled(len, T::ZERO, quantity);
        Ok(Self {
            backend: shape.backend.clone(),
            layout: shape.layout,
            blocks: shape.blocks,
            provided: shape.provided,
            lin: Linearization {
                n,
                stages: s,
                full: false,
                jacobians,
                points: filled(points, scioxide_core::quantity!("stage_points"))?,
                rhs: filled(points, scioxide_core::quantity!("stage_point_rhs"))?,
                times: try_filled(
                    shape.blocks,
                    S::ZERO,
                    scioxide_core::quantity!("stage_times"),
                )?,
                floors: try_filled(n, S::ONE, scioxide_core::quantity!("difference_floors"))?,
                states: filled(size, scioxide_core::quantity!("stage_states"))?,
                mass,
                strategy: super::DEFAULT_STRATEGY,
                time: S::ZERO,
                step: S::ZERO,
                coefficients: try_filled(
                    checked_mul(s, s, scioxide_core::quantity!("coupling"))?,
                    S::ZERO,
                    scioxide_core::quantity!("coupling"),
                )?,
                nodes: try_filled(s, S::ZERO, scioxide_core::quantity!("coupling_nodes"))?,
            },
            linear,
            differences: Differences::new(n)?,
            mass_factor,
            mass_factor_valid: false,
            cache: Cache {
                valid: false,
                strategy: None,
                time: None,
                constant: false,
                revision: 0,
                factor: None,
                mass: None,
                refresh: false,
                accepted: 0,
            },
            stats: CacheStats::default(),
            exact_jacobian: false,
            lagged_jacobian: false,
            lagged_factor: false,
            contraction: None,
            factor_coefficients: try_filled(
                s * s,
                S::ZERO,
                scioxide_core::quantity!("factor_coupling"),
            )?,
            product: filled(size, scioxide_core::quantity!("stage_product"))?,
            scratch: filled(
                operator_scratch.max(n),
                scioxide_core::quantity!("stage_scratch"),
            )?,
            krylov: SolveState::new(),
            evaluation: SolveState::new(),
        })
    }

    /// Whether this storage serves `n` unknowns with `shape`.
    pub(crate) fn matches(&self, n: usize, shape: KernelShape<'_>) -> bool {
        self.lin.n == n
            && self.lin.stages == shape.stages
            && self.blocks == shape.blocks
            && self.provided == shape.provided
            && self.layout == shape.layout
            && self.backend == *shape.backend
    }

    /// The requested shape this kernel was prepared for.
    pub(crate) const fn shape(&self) -> KernelShape<'_> {
        KernelShape {
            backend: &self.backend,
            layout: self.layout,
            stages: self.lin.stages,
            blocks: self.blocks,
            provided: self.provided,
        }
    }

    /// Distinct Jacobian blocks prepared for full Newton.
    pub(crate) const fn blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// A supplied mass matrix; `None` represents identity on every backend.
    pub(crate) const fn mass(&self) -> Option<&Matrix<T>> {
        if matches!(self.cache.mass, Some(MassKey::Identity)) {
            None
        } else {
            self.lin.mass.as_ref()
        }
    }

    pub(crate) fn invalidate_factor(&mut self) -> Result<(), SciOxideError> {
        if self.cache.factor.is_some() {
            count(
                &mut self.stats.factor_invalidations,
                scioxide_core::quantity!("ode_factor_invalidation_counter"),
            )?;
            self.cache.factor = None;
        }
        Ok(())
    }

    /// Forget the raw Jacobian and its factor, counting each valid entry once.
    pub(crate) fn invalidate_jacobian(&mut self) -> Result<(), SciOxideError> {
        let mut stats = self.stats;
        if self.cache.valid {
            count(
                &mut stats.jacobian_invalidations,
                scioxide_core::quantity!("ode_jacobian_invalidation_counter"),
            )?;
        }
        if self.cache.factor.is_some() {
            count(
                &mut stats.factor_invalidations,
                scioxide_core::quantity!("ode_factor_invalidation_counter"),
            )?;
        }
        self.stats = stats;
        self.cache.factor = None;
        self.cache.valid = false;
        self.cache.strategy = None;
        self.cache.time = None;
        self.cache.constant = false;
        self.cache.refresh = false;
        Ok(())
    }

    /// Discard every numerical value while keeping storage for a new epoch.
    pub(crate) fn invalidate(&mut self) -> Result<(), SciOxideError> {
        self.invalidate_jacobian()?;
        self.contraction = None;
        self.cache.mass = None;
        self.mass_factor_valid = false;
        if let Jacobians::Operator { sources, .. } = &self.lin.jacobians {
            sources.iter().for_each(|source| source.set(None));
        }
        Ok(())
    }

    /// Initialize identity mass after invalidating the epoch, without a provider query.
    pub(crate) fn initialize_identity_mass(&mut self) {
        debug_assert!(self.cache.mass.is_none());
        if let Some(mass) = &mut self.lin.mass {
            mass.as_mut_slice().fill(T::ZERO);
            (0..self.lin.n).for_each(|i| mass.as_mut_slice()[i * self.lin.n + i] = T::ONE);
        }
        self.cache.mass = Some(MassKey::Identity);
    }

    /// Load the step's mass matrix, reusing a revision loaded in this epoch.
    pub(crate) fn load_mass<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<MassKey, SciOxideError> {
        let key = system
            .mass_matrix_revision()
            .map_or(MassKey::Identity, MassKey::Revision);
        if self.lin.mass.is_none() && key != MassKey::Identity {
            return Err(invalid(
                scioxide_core::argument!("system"),
                "sparse and Krylov stages require identity mass",
            ));
        }
        if self.cache.mass == Some(key) {
            return Ok(key);
        }
        self.invalidate_factor()?;
        self.mass_factor_valid = false;
        self.cache.mass = None;
        let n = self.lin.n;
        if let Some(mass) = &mut self.lin.mass {
            mass.as_mut_slice().fill(T::ZERO);
            match key {
                MassKey::Identity => {
                    (0..n).for_each(|i| mass.as_mut_slice()[i * n + i] = T::ONE);
                }
                MassKey::Revision(_) => {
                    crate::stepper::invoke(EvaluationRole::LinearOperator, request, cx, |cx| {
                        system.mass_matrix(&mut mass.view_mut(), cx)
                    })?;
                    crate::traits::validate_provider_matrix(mass, n)?;
                }
            }
        }
        self.cache.mass = Some(key);
        Ok(key)
    }

    /// Linearize block `block` at `(time, state)` with `f(state) = base`; simplified block
    /// zero reuses a valid Jacobian unless `force` is set. Finite-difference steps are
    /// floored by the request's absolute tolerances.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linearize<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        strategy: DerivativeStrategy,
        block: usize,
        time: S,
        state: &[T],
        base: &[T],
        force: bool,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.lin.n;
        crate::safety::validate_finite_time(time, scioxide_core::NonFiniteOrigin::Input)?;
        if block >= self.blocks || state.len() != n || base.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, n),
                scioxide_core::Dimensions::two(state.len(), base.len()),
            ));
        }
        finite(state, scioxide_core::NonFiniteOrigin::Input)?;
        let movement_limit = reuse::movement_limit::<S>()?;

        if !force
            && block == 0
            && self.cache.valid
            && self.cache.strategy == Some(strategy)
            && ((self.cache.constant && system.jacobian_is_constant())
                || (!self.cache.refresh
                    && self
                        .cache
                        .time
                        .is_some_and(|old| reuse::movement(old, time) <= movement_limit)
                    && reuse::within_movement(&self.lin.points[..n], state, movement_limit)))
        {
            self.lagged_jacobian = !self.cache.constant
                && (self.cache.time != Some(time) || self.lin.points[..n] != *state);
            return count(
                &mut self.stats.jacobian_hits,
                scioxide_core::quantity!("ode_jacobian_cache_hit_counter"),
            );
        }
        self.lagged_jacobian = false;
        self.invalidate_jacobian()?;
        let default = scioxide_core::ErrorTolerance::default();
        let tolerance = request.error_tolerance.unwrap_or(&default);
        for (index, floor) in self.lin.floors.iter_mut().enumerate() {
            let absolute = tolerance.threshold(index, n, S::ZERO)?;
            *floor = if absolute > S::ZERO { absolute } else { S::ONE };
        }
        let range = block * n..(block + 1) * n;
        self.lin.points[range.clone()].copy_from_slice(state);
        self.lin.rhs[range].copy_from_slice(base);
        self.lin.times[block] = time;
        self.lin.strategy = strategy;
        let exact = match &mut self.lin.jacobians {
            Jacobians::Dense(blocks) => {
                let source = super::jacobian::dense(
                    system,
                    strategy,
                    time,
                    state,
                    base,
                    &self.lin.floors,
                    &mut blocks[block],
                    &mut self.differences,
                    request,
                    cx,
                )?;
                !matches!(source, DerivativeCandidate::FiniteDifference(_))
            }
            Jacobians::Sparse { columns, values } => {
                let nnz = columns.len();
                let source = super::jacobian::sparse(
                    system,
                    strategy,
                    time,
                    state,
                    base,
                    &self.lin.floors,
                    columns,
                    &mut values[block * nnz..(block + 1) * nnz],
                    &mut self.differences,
                    request,
                    cx,
                )?;
                !matches!(source, DerivativeCandidate::FiniteDifference(_))
            }
            Jacobians::Operator { sources, .. } => {
                sources[block].set(None);
                false
            }
        };
        self.exact_jacobian = exact && (block == 0 || self.exact_jacobian);

        self.cache.revision = self.cache.revision.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("ode_jacobian_revision_counter"))
        })?;
        if block == 0 {
            self.cache.accepted = 0;
            self.cache.valid = true;
            self.cache.strategy = Some(strategy);
            self.cache.time = Some(time);
            self.cache.constant = system.jacobian_is_constant();
        }
        Ok(())
    }

    pub(crate) fn requires_rate(&self) -> bool {
        self.lagged_jacobian || self.lagged_factor
    }

    /// The contraction rate last measured by a Newton solve of this epoch.
    pub(crate) const fn contraction(&self) -> Option<S> {
        self.contraction
    }

    /// Carry a measured contraction rate to the next solve, converged or not.
    pub(crate) fn observe_contraction(&mut self, rate: Option<S>) {
        if rate.is_some() {
            self.contraction = rate;
        }
    }

    /// A current native direct factor built from exact Jacobian callbacks.
    pub(crate) fn affine_direct(&self) -> bool {
        self.exact_jacobian
            && self.cache.factor.is_some()
            && self.layout == StageLayout::Coupled
            && !self.provided
            && !self.requires_rate()
            && matches!(
                self.backend,
                OdeStageBackend::Dense | OdeStageBackend::Sparse { .. }
            )
    }

    /// Schedule refresh after slow convergence, keeping this step's estimator valid.
    pub(crate) fn observe_newton(
        &mut self,
        report: super::newton::NewtonReport<S>,
    ) -> Result<(), SciOxideError> {
        if reuse::slow(report)? {
            if self.lagged_factor {
                self.invalidate_factor()?;
            } else {
                self.cache.refresh = true;
            }
        }
        Ok(())
    }

    /// W methods have no Newton-rate observation: bound a lagged approximation's age.
    pub(crate) fn accept_approximation(&mut self, maximum_age: usize) {
        self.cache.accepted = self.cache.accepted.saturating_add(1);
        if self.cache.accepted >= maximum_age {
            self.cache.refresh = true;
        }
    }

    /// Rejection keeps problem-level data. A nonconstant W approximation is refreshed.
    pub(crate) fn reject(&mut self) -> Result<(), SciOxideError> {
        self.invalidate_factor()
    }

    pub(crate) fn reject_approximation(&mut self) -> Result<(), SciOxideError> {
        self.cache.refresh = true;
        self.reject()
    }

    /// Simplified Newton may use a nearby effective factor as a chord iteration.
    pub(crate) fn factor_newton(
        &mut self,
        coupling: Coupling<'_, S>,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        // Exact affine solves have no contraction iteration to absorb a changed
        // effective coefficient. Rebuild their factor whenever the key changes.
        let approximate = !(self.cache.constant && self.exact_jacobian);
        self.factor_with_reuse(coupling, false, approximate, request, cx)
    }

    /// Prepare `E` for `coupling`; `full` uses one Jacobian per stage block.
    pub(crate) fn factor(
        &mut self,
        coupling: Coupling<'_, S>,
        full: bool,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.factor_with_reuse(coupling, full, false, request, cx)
    }

    fn factor_with_reuse(
        &mut self,
        coupling: Coupling<'_, S>,
        full: bool,
        approximate: bool,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let s = self.lin.stages;
        if coupling.coefficients.len() != s * s
            || coupling.nodes.len() != s
            || (full && self.blocks != s)
        {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "stage coupling does not match the prepared blocks",
            ));
        }
        let mass = self.cache.mass.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "load the mass matrix before factoring",
            )
        })?;
        let key = FactorKey {
            revision: self.cache.revision,
            mass,
            step: coupling.step,
            full,
        };
        let n = self.lin.n;
        self.lin.time = coupling.time;
        self.lin.nodes.copy_from_slice(coupling.nodes);
        for block in 0..s {
            let source = if full { block } else { 0 } * n;
            self.lin.states[block * n..(block + 1) * n]
                .copy_from_slice(&self.lin.points[source..source + n]);
        }
        let exact =
            self.cache.factor == Some(key) && self.factor_coefficients == coupling.coefficients;
        let nearby_ratio = self.cache.factor.and_then(|old| {
            if old.revision != key.revision || old.mass != key.mass || old.full {
                return None;
            }
            if s == 1 {
                Some(
                    coupling.step / old.step
                        * (coupling.coefficients[0] / self.factor_coefficients[0]),
                )
            } else if self.factor_coefficients == coupling.coefficients {
                Some(coupling.step / old.step)
            } else {
                None
            }
        });
        let nearby = approximate
            && !full
            && !self.provided
            && self.layout == StageLayout::Coupled
            && matches!(&self.linear, EffectiveStorage::Coupled(linear) if !linear.is_iterative())
            && nearby_ratio
                .map(reuse::nearby)
                .transpose()?
                .unwrap_or(false);
        self.lin.step = coupling.step;
        self.lin.full = full;
        self.lin.coefficients.copy_from_slice(coupling.coefficients);
        self.lagged_factor = nearby && !exact;
        if self.cache.valid && (exact || nearby) {
            return count(
                &mut self.stats.factor_hits,
                scioxide_core::quantity!("ode_factor_cache_hit_counter"),
            );
        }
        self.invalidate_factor()?;
        self.lin.step = coupling.step;
        self.lin.full = full;
        self.lin.coefficients.copy_from_slice(coupling.coefficients);
        match &mut self.linear {
            EffectiveStorage::Radau(radau) => radau.factor(&self.lin, request, cx)?,
            EffectiveStorage::Coupled(linear) if !linear.is_iterative() => {
                assemble(&self.lin, linear)?;
                admit_provider(cx, EvaluationRole::Factorization, request.trace)?;
                linear.factor(cx.execution())?;
            }
            _ => {}
        }
        self.factor_coefficients
            .copy_from_slice(coupling.coefficients);
        self.cache.factor = Some(key);
        Ok(())
    }

    /// The effective operator `E` at the current linearization.
    pub(crate) fn operator<'a, P: OdeSystem<S, T> + ?Sized>(
        &'a self,
        system: &'a P,
        request: StepRequest<'a, S, T>,
    ) -> Effective<'a, S, T, P> {
        Effective {
            lin: &self.lin,
            system,
            request,
        }
    }

    /// Solve `E·correction = residual` with the prepared backend or an injected provider.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        request: StepRequest<'_, S, T>,
        accuracy: StageAccuracy<'_, S, T>,
        state: &[T],
        residual: &[T],
        correction: &mut [T],
        provider: Option<&dyn StageLinearSolver<S, T>>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let size = self.lin.n * self.lin.stages;
        if residual.len() != size || correction.len() != size || state.len() != size {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "stage right-hand side shape changed",
            ));
        }
        if let EffectiveStorage::Radau(radau) = &mut self.linear
            && !self.provided
        {
            let relative = if matches!(self.backend, OdeStageBackend::Krylov { .. }) {
                iterative_tolerance(accuracy, state, residual)?
            } else {
                S::ZERO
            };
            radau.solve(
                &self.lin,
                system,
                &self.backend,
                request,
                relative,
                residual,
                correction,
                &mut self.krylov,
                cx,
            )?;
            if matches!(self.backend, OdeStageBackend::Krylov { .. }) {
                let operator = Effective {
                    lin: &self.lin,
                    system,
                    request,
                };
                let outcome = verify(
                    &operator,
                    relative,
                    residual,
                    correction,
                    &mut self.product,
                    &mut self.scratch,
                    &mut self.evaluation,
                    cx,
                );

                outcome?;
            }
            return Ok(());
        }
        admit_provider(cx, EvaluationRole::LinearSolve, request.trace)?;

        let Self {
            lin,
            linear,
            krylov,
            evaluation,
            product,
            scratch,
            ..
        } = self;
        let lin = &*lin;
        let operator = Effective {
            lin,
            system,
            request,
        };
        match (provider, linear) {
            (Some(provider), _) => {
                provider
                    .solve_into(&operator, state, residual, correction, accuracy, cx)
                    .inspect_err(|_| accuracy.mark_provider_failure())?;
                if correction
                    .iter()
                    .any(|value| !value.has_finite_components())
                {
                    accuracy.mark_provider_failure();
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Callback,
                        "non-finite provider correction",
                    ));
                }
                verify(
                    &operator,
                    iterative_tolerance(accuracy, state, residual)?,
                    residual,
                    correction,
                    product,
                    scratch,
                    evaluation,
                    cx,
                )
            }
            (None, EffectiveStorage::Coupled(linear)) if linear.is_iterative() => {
                let relative = iterative_tolerance(accuracy, state, residual)?;
                let preconditioner = preconditioner(lin, system, request, cx)?;
                let identity = scioxide_core::SolveIdentity::new(
                    ProblemToken::new(0),
                    scioxide_core::algorithm_id!("ode", "stage-gmres", 1),
                    size as u128,
                    0,
                );
                SolveScope::from(&mut *cx).run(
                    krylov,
                    identity,
                    SolveBudgets::unbounded(),
                    |context| {
                        linear
                            .solve_iterative(
                                &operator,
                                preconditioner,
                                residual,
                                correction,
                                relative,
                                request.inner_limit.get(),
                                context,
                            )
                            .map(|_| ())
                    },
                )?;
                finite(correction, scioxide_core::NonFiniteOrigin::Callback)?;
                verify(
                    &operator, relative, residual, correction, product, scratch, evaluation, cx,
                )
            }
            (None, EffectiveStorage::Coupled(linear)) => linear
                .solve_direct(residual, correction, cx.execution())
                .and_then(|()| finite(correction, scioxide_core::NonFiniteOrigin::Callback)),
            (None, _) => Err(invalid(
                scioxide_core::argument!("workspace"),
                "an injected linear provider was configured for this storage",
            )),
        }
    }

    /// Radau's real-mode reciprocal eigenvalue, sharing the stage normalization.
    pub(crate) fn radau_shift(&self) -> Result<S, SciOxideError> {
        match &self.linear {
            EffectiveStorage::Radau(radau) => Ok(radau.real_shift()),
            _ => Err(invalid(
                scioxide_core::argument!("workspace"),
                "Radau storage required",
            )),
        }
    }

    /// Reuse the real shifted factor for the embedded estimator, including injected stages.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn radau_filter<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        request: StepRequest<'_, S, T>,
        accuracy: StageAccuracy<'_, S, T>,
        state: &[T],
        rhs: &[T],
        error: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let relative = if matches!(self.backend, OdeStageBackend::Krylov { .. }) {
            iterative_tolerance(accuracy, state, rhs)?
        } else {
            S::ZERO
        };
        let EffectiveStorage::Radau(radau) = &mut self.linear else {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "Radau storage required",
            ));
        };
        radau.filter(
            &self.lin,
            system,
            &self.backend,
            request,
            relative,
            rhs,
            error,
            &mut self.krylov,
            cx,
        )
    }

    /// `output = M⁻¹·rhs`, factoring a revisioned mass matrix once per revision.
    pub(crate) fn solve_mass(
        &mut self,
        request: StepRequest<'_, S, T>,
        rhs: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let identity = matches!(self.cache.mass, Some(MassKey::Identity));
        let Some(mass) = self.lin.mass.as_ref().filter(|_| !identity) else {
            output.copy_from_slice(rhs);
            return Ok(());
        };
        if !self.mass_factor_valid {
            admit_provider(cx, EvaluationRole::Factorization, request.trace)?;
            self.mass_factor.factor(mass.view(), cx.execution())?;
            self.mass_factor_valid = true;
        }
        admit_provider(cx, EvaluationRole::LinearSolve, request.trace)?;
        self.mass_factor.solve_into(rhs, output, cx.execution())?;
        Ok(())
    }

    /// `output = J₀·input` for the first linearization block.
    /// Diagnostic multiplication never acquires or refreshes a linearization and
    /// never falls through to an operator's finite-difference products.
    pub(crate) fn cached_product(
        &self,
        time: S,
        state: &[T],
        input: &[T],
        output: &mut [T],
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        if !self.cache.valid
            || self.cache.refresh
            || matches!(self.lin.jacobians, Jacobians::Operator { .. })
        {
            return Ok(false);
        }
        let limit = reuse::movement_limit()?;
        if !self.cache.constant
            && (!self
                .cache
                .time
                .is_some_and(|old| reuse::movement(old, time) <= limit)
                || !reuse::within_movement(&self.lin.points[..self.lin.n], state, limit))
        {
            return Ok(false);
        }
        admit_provider(cx, EvaluationRole::LinearOperator, request.trace)?;
        match &self.lin.jacobians {
            Jacobians::Dense(blocks) => {
                for (out, row) in output
                    .iter_mut()
                    .zip(blocks[0].as_slice().chunks(self.lin.n.max(1)))
                {
                    *out = row
                        .iter()
                        .zip(input)
                        .fold(T::ZERO, |sum, (&a, &v)| sum + a * v);
                }
                finite(output, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
            Jacobians::Sparse { columns, values } => {
                columns.multiply(&values[..columns.len()], input, output)?
            }
            Jacobians::Operator { .. } => return Ok(false),
        }
        Ok(true)
    }

    /// Scalar scratch configured for an analytic Jacobian product.
    pub(crate) fn diagnostic_scratch(&self) -> usize {
        match self.backend {
            OdeStageBackend::Krylov {
                operator_scratch, ..
            } => operator_scratch,
            _ => 0,
        }
    }

    /// `output = J₀·input` for the first linearization block.
    pub(crate) fn apply_jacobian<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        request: StepRequest<'_, S, T>,
        input: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.lin.n;
        if input.len() != n || output.len() != n {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "Jacobian product shape changed",
            ));
        }
        admit_provider(cx, EvaluationRole::LinearOperator, request.trace)?;
        match &self.lin.jacobians {
            Jacobians::Dense(blocks) => {
                // Acquired Jacobians are already finite. Reuse the shared native
                // dot kernel, replaying exceptional rows without materializing
                // a general linear-operator application or rescanning the matrix.
                for (out, row) in output.iter_mut().zip(blocks[0].as_slice().chunks(n.max(1))) {
                    let products = row.iter().copied().zip(input.iter().copied());
                    *out = match native_dot(products.clone()) {
                        Some(value) => value,
                        None => scioxide_core::backend::standard::sum_products(
                            products,
                            scioxide_core::Accuracy::Standard,
                        )?,
                    };
                }
            }
            Jacobians::Sparse { columns, values } => {
                columns.multiply(&values[..columns.len()], input, output)?;
            }
            Jacobians::Operator { sources, scratch } => {
                let lin = &self.lin;
                let point = ProductPoint {
                    time: lin.times[0],
                    state: &lin.points[..n],
                    base: &lin.rhs[..n],
                    floors: &lin.floors,
                };
                let workspace = &mut self.scratch[..*scratch + 3 * n];

                SolveScope::from(&mut *cx).run(
                    &mut self.evaluation,
                    product_identity(n),
                    SolveBudgets::unbounded(),
                    |context| {
                        super::jacobian::product(
                            system,
                            lin.strategy,
                            &sources[0],
                            point,
                            input,
                            output,
                            workspace,
                            request,
                            context,
                        )?;
                        Ok(())
                    },
                )?;
            }
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}

/// Native dot product of finite operands, or `None` when it may have lost range: a
/// non-finite sum, or a component below the normal range unless every product has an
/// exact zero operand, as in structurally zero rows and zero inputs.
fn native_dot<T: NumericScalar>(mut products: impl Iterator<Item = (T, T)> + Clone) -> Option<T> {
    let value = scioxide_core::backend::standard::dot(products.clone());
    let normal = |component: T::Real| component.abs() >= T::Real::MIN_POSITIVE;
    let ordinary = value.has_finite_components()
        && normal(value.real_component())
        && (T::IS_REAL || normal(value.imaginary_component()));
    (ordinary
        || (value == T::ZERO && products.all(|(left, right)| left == T::ZERO || right == T::ZERO)))
    .then_some(value)
}

/// Identity of one nested product epoch; epochs are scoped by their parent solve.
fn product_identity(n: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("ode", "jacobian-product", 1),
        n as u128,
        0,
    )
}

/// Apply `E` to an iterative correction and require its true defect within `relative`.
#[allow(clippy::too_many_arguments)]
fn verify<S, T, P>(
    operator: &Effective<'_, S, T, P>,
    relative: S,
    residual: &[T],
    correction: &[T],
    product: &mut [T],
    scratch: &mut [T],
    evaluation: &mut SolveState,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    admit_provider(cx, EvaluationRole::LinearOperator, operator.request.trace)?;
    let scratch = &mut scratch[..operator.apply_workspace_len()?];
    SolveScope::from(&mut *cx).run(
        evaluation,
        product_identity(product.len()),
        SolveBudgets::unbounded(),
        |context| {
            operator
                .apply_into_scoped(
                    correction,
                    product,
                    scratch,
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)
        },
    )?;
    verify_defect(product, residual, relative).map(|_| ())
}

fn preconditioner<'a, S, T, P>(
    lin: &Linearization<S, T>,
    system: &'a P,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<Option<&'a dyn Preconditioner<T>>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    request.trace.provider(if lin.stages == 1 {
        system.stage_preconditioner(
            lin.times[0],
            &lin.points[..lin.n],
            lin.step * lin.coefficients[0],
            cx,
        )
    } else {
        system.coupled_preconditioner(
            lin.time,
            &lin.states,
            lin.step,
            &lin.coefficients,
            &lin.nodes,
            cx,
        )
    })
}

/// Assemble `E` into dense or sparse storage.
fn assemble<S, T>(
    lin: &Linearization<S, T>,
    linear: &mut LinearStorage<T>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let (n, s) = (lin.n, lin.stages);
    let block = |column: usize| if lin.full { column } else { 0 };
    match &lin.jacobians {
        Jacobians::Dense(blocks) => {
            let size = n * s;
            let matrix = linear.dense_mut()?.as_mut_slice();
            for block_row in 0..s {
                for block_column in 0..s {
                    let jacobian = blocks[block(block_column)].as_slice();
                    let coefficient = lin.coefficients[block_row * s + block_column];
                    for row in 0..n {
                        for column in 0..n {
                            let mass = match (&lin.mass, block_row == block_column) {
                                (Some(mass), true) => mass.as_slice()[row * n + column],
                                (None, true) if row == column => T::ONE,
                                _ => T::ZERO,
                            };
                            matrix[(block_row * n + row) * size + block_column * n + column] =
                                crate::safety::stage_affine(
                                    mass,
                                    -lin.step,
                                    [(coefficient, jacobian[row * n + column])],
                                )?;
                        }
                    }
                }
            }
            Ok(())
        }
        Jacobians::Sparse { columns, values } => {
            let p = columns.pattern.view();
            let nnz = columns.len();
            linear.assemble_sparse(|_, output| {
                let mut next = 0;
                for block_row in 0..s {
                    for row in 0..n {
                        for block_column in 0..s {
                            let shift = lin.step * lin.coefficients[block_row * s + block_column];
                            let offset = block(block_column) * nnz;
                            for k in p.row_ptr()[row]..p.row_ptr()[row + 1] {
                                let identity =
                                    block_row == block_column && p.col_indices()[k] == row;
                                output[next] = crate::safety::state_increment(
                                    if identity { T::ONE } else { T::ZERO },
                                    -shift,
                                    values[offset + k],
                                )?;
                                next += 1;
                            }
                        }
                    }
                }
                Ok(())
            })
        }
        Jacobians::Operator { .. } => Err(invalid(
            scioxide_core::argument!("workspace"),
            "operator Jacobians are applied matrix-free",
        )),
    }
}

/// Matrix-free effective operator `E = I_s ⊗ M − H ⊗ J` at a kernel's linearization.
pub(crate) struct Effective<'a, S: RealScalar, T: NumericScalar<Real = S>, P: ?Sized> {
    lin: &'a Linearization<S, T>,
    system: &'a P,
    request: StepRequest<'a, S, T>,
}

impl<S, T, P> Effective<'_, S, T, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn apply(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let lin = self.lin;
        let (n, s) = (lin.n, lin.stages);
        if input.len() != n * s || output.len() != n * s {
            return Err(invalid(
                scioxide_core::argument!("input"),
                "effective operator shape mismatch",
            ));
        }
        let block = |column: usize| if lin.full { column } else { 0 };
        let shift =
            |row: usize, column: usize| T::from_real(lin.step * lin.coefficients[row * s + column]);
        match &lin.jacobians {
            Jacobians::Dense(blocks) => {
                for block_row in 0..s {
                    for row in 0..n {
                        let x = &input[block_row * n..(block_row + 1) * n];
                        let mut value = match &lin.mass {
                            Some(mass) => mass.as_slice()[row * n..(row + 1) * n]
                                .iter()
                                .zip(x)
                                .fold(T::ZERO, |sum, (&m, &v)| sum + m * v),
                            None => x[row],
                        };
                        for block_column in 0..s {
                            let jacobian =
                                &blocks[block(block_column)].as_slice()[row * n..(row + 1) * n];
                            let product = jacobian
                                .iter()
                                .zip(&input[block_column * n..(block_column + 1) * n])
                                .fold(T::ZERO, |sum, (&j, &v)| sum + j * v);
                            value = value - shift(block_row, block_column) * product;
                        }
                        output[block_row * n + row] = value;
                    }
                }
            }
            Jacobians::Sparse { columns, values } => {
                let p = columns.pattern.view();
                let nnz = columns.len();
                for block_row in 0..s {
                    for row in 0..n {
                        let mut value = input[block_row * n + row];
                        for block_column in 0..s {
                            let offset = block(block_column) * nnz;
                            let product =
                                (p.row_ptr()[row]..p.row_ptr()[row + 1]).fold(T::ZERO, |sum, k| {
                                    sum + values[offset + k]
                                        * input[block_column * n + p.col_indices()[k]]
                                });
                            value = value - shift(block_row, block_column) * product;
                        }
                        output[block_row * n + row] = value;
                    }
                }
            }
            Jacobians::Operator {
                sources,
                scratch: provider,
            } => {
                // Products evaluate system callbacks, which need the invoking scope.
                let Some(context) = context else {
                    return Err(invalid(
                        scioxide_core::argument!("operator"),
                        "matrix-free stage operators require scoped application",
                    ));
                };
                let (products, rest) = scratch.split_at_mut(n * s);
                let workspace = &mut rest[..*provider + 3 * n];
                for column in 0..s {
                    let b = block(column);
                    let point = ProductPoint {
                        time: lin.times[b],
                        state: &lin.points[b * n..(b + 1) * n],
                        base: &lin.rhs[b * n..(b + 1) * n],
                        floors: &lin.floors,
                    };
                    super::jacobian::product(
                        self.system,
                        lin.strategy,
                        &sources[b],
                        point,
                        &input[column * n..(column + 1) * n],
                        &mut products[column * n..(column + 1) * n],
                        workspace,
                        self.request,
                        &mut *context,
                    )?;
                }
                for block_row in 0..s {
                    for row in 0..n {
                        output[block_row * n + row] =
                            (0..s).fold(input[block_row * n + row], |value, column| {
                                value - shift(block_row, column) * products[column * n + row]
                            });
                    }
                }
            }
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}

impl<S, T, P> LinearOperator<T> for Effective<'_, S, T, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn rows(&self) -> usize {
        self.lin.n * self.lin.stages
    }
    fn cols(&self) -> usize {
        self.lin.n * self.lin.stages
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(match &self.lin.jacobians {
            Jacobians::Operator { scratch, .. } => {
                scratch + self.lin.n * self.lin.stages + 3 * self.lin.n
            }
            _ => 0,
        })
    }
    fn apply_into(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, scratch, None)
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(input, output, scratch, Some(&mut **context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> core::fmt::Debug
    for StageKernel<S, T>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StageKernel")
            .field("dimension", &self.lin.n)
            .field("shape", &self.shape())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stepper::{StepTrace, testing::with_context};

    #[test]
    fn zero_operand_rows_stay_native_and_underflow_replays() {
        let tiny = f64::from_bits(1);
        let dot =
            |row: &[f64], input: &[f64]| native_dot(row.iter().copied().zip(input.iter().copied()));
        // Structurally zero rows and zero inputs are exact.
        assert_eq!(dot(&[0., 0., 0.], &[1., -2., 3.]), Some(0.));
        assert_eq!(dot(&[1., -2., 3.], &[0., 0., 0.]), Some(0.));
        assert_eq!(dot(&[0., 4., 0.], &[5., 0., 0.]), Some(0.));
        assert_eq!(dot(&[2., 0.5], &[0.25, 4.]), Some(2.5));
        // Products that underflow to zero, or cancel exactly, replay.
        assert_eq!(dot(&[tiny, tiny], &[0.5, 0.5]), None);
        assert_eq!(dot(&[1., -1.], &[1., 1.]), None);
        assert_eq!(dot(&[f64::MAX, f64::MAX], &[2., -2.]), None);
        let complex = |re, im| scioxide_core::Complex::new(re, im);
        let zero = complex(0., 0.);
        assert_eq!(
            native_dot([(zero, complex(1., 2.)), (complex(3., -1.), zero)].into_iter()),
            Some(zero)
        );
        assert_eq!(
            native_dot([(complex(tiny, 0.), complex(0.5, 0.))].into_iter()),
            None
        );
    }

    #[test]
    fn assembled_products_replay_overflow_and_underflow_with_reused_storage() {
        let mut kernel = StageKernel::<f64, f64>::new(
            2,
            KernelShape {
                backend: &OdeStageBackend::Dense,
                layout: StageLayout::Coupled,
                stages: 1,
                blocks: 1,
                provided: false,
            },
        )
        .unwrap();
        let tiny = f64::from_bits(1);
        if let Jacobians::Dense(blocks) = &mut kernel.lin.jacobians {
            blocks[0]
                .as_mut_slice()
                .copy_from_slice(&[f64::MAX, -f64::MAX, tiny, tiny]);
        }
        let system = crate::OdeFn::new(2, |_: f64, _: &[f64], out: &mut [f64]| {
            out.fill(0.);
            Ok(())
        })
        .unwrap();
        let trace = StepTrace::default();
        let mut output = [0.; 2];
        with_context(|cx| {
            for (direction, expected) in
                [([2.; 2], [0., f64::from_bits(4)]), ([0.5; 2], [0., tiny])]
            {
                {
                    kernel
                        .apply_jacobian(
                            &system,
                            StepRequest::new(&trace),
                            &direction,
                            &mut output,
                            cx,
                        )
                        .unwrap();
                };
                assert_eq!(output, expected);
            }
            Ok(())
        })
        .unwrap();
    }
}
