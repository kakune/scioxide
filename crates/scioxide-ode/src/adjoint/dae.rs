use super::checkpoint::{Kernel, Step as RecordedStep, Tape, TapePlan, TapeShape, Trial};
type Step<S> = RecordedStep<S, BdfStep<S>>;
use super::dae_linear::{Transpose, prepare_backend, storage_bound};
use super::*;
use crate::dae::bdf::{BdfStep, Formula, MAX_ORDER};
use crate::dae::{DaeStorage, Unaccepted};
use crate::multistep::polynomial::{Polynomial, weights};
use crate::shell::Shell;
use crate::{
    DaeBdf, DaeInitialization, DaeNumericalOptions, DaeOptions, DaePartition, DaeReport,
    DaeTermination, OdeStageBackend,
};
use core::cell::RefCell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{AttemptRecord, Capacity, ProblemToken, RelativeTolerance, SolveState};

// Checkpointed discrete/continuous adjoints for regular residual index-1 DAEs.
//
// The forward method of [`Self::new`] is residual BDF with adaptive orders one through
// five; every history state in an accepted equation participates in its discrete
// derivative. [`Self::backward_euler`] is its order-one configuration and
// [`Self::trapezoid`] the fixed-step implicit trapezoid, whose discrete adjoint also
// carries the cotangent of the recorded rate. Initial gradients use free differential
// coordinates in partition order; algebraic guesses are not independent coordinates.
// Events and higher-index systems are unsupported. The plan takes `&self`, is `Clone`
// and holds no caller identity.
//
// The common adjoint controls own the forward step and solve budgets. Shared
// DAE numerical controls supply error, residual, initialization and iteration policies.
//
// [`Self::solve_into`] is the atomic execution form; [`Self::solve`] is the owned
// convenience.
//

#[derive(Debug, Clone)]
#[must_use]
pub(super) struct PreparedDaeAdjoint<T: NumericScalar> {
    pub(super) tape: TapePlan,
    observations: Vec<T::Real>,
    options: Policies<T::Real>,
    linear_residual: RelativeTolerance<T::Real>,
    primal: (DaeBdf, DaeOptions<T::Real>),
    projection: (DaeBdf, DaeOptions<T::Real>),
    partition: DaePartition,
    backend: OdeStageBackend,
    transpose_backend: OdeStageBackend,
    permutation: Vec<usize>,
    provider_scratch: usize,
    objective_scratch: usize,
    differential: usize,
    formula: Formula,
}

// Storage one [`PreparedDaeAdjoint`] needs, from
// [`PreparedDaeAdjoint::workspace_requirements`].
//
// A workspace serves exactly one requirement shape and stage backend at a time;
// reserving another replaces its storage.
#[derive(Debug, Clone)]
pub(super) struct DaeStorageRequirements<T: NumericScalar> {
    tape: TapeShape,
    primal: (DaeBdf, DaeOptions<T::Real>),
    transpose: OdeStageBackend,
    provider: usize,
    objective: usize,
    differential: usize,
    bytes: usize,
}

impl<T: NumericScalar> DaeStorageRequirements<T> {
    // Length of the initial-gradient output: the free differential coordinates.
    pub(super) const fn initial_gradient_len(&self) -> usize {
        self.differential
    }
    // Length of the parameter-gradient output.
    pub(super) const fn parameter_gradient_len(&self) -> usize {
        self.tape.parameters()
    }
    // Admitted conservative upper bound on the numerical heap payload.
    pub(super) const fn workspace_bytes(&self) -> usize {
        self.bytes
    }
}

// Reusable DAE trajectory, projection, transpose-solve and checkpoint storage, solve
// state and last attempt.
//
// [`DaeAdjointBuffers::new`] allocates nothing; [`PreparedDaeAdjoint::workspace`] or
// [`Self::try_reserve`] prepare it.
pub(super) struct DaeAdjointBuffers<T: NumericScalar> {
    pub(super) storage: Option<DaeAdjointStorage<T>>,
    pub(super) shell: Shell<AdjointReport<T::Real>>,
}

pub(super) struct DaeAdjointStorage<T: NumericScalar> {
    pub(super) tape: Tape<T, BdfStep<T::Real>>,
    work: DaeWork<T>,
}

struct LinearWork<T: NumericScalar> {
    solver: crate::dae::linear::DaeLinear<T>,
    products: Products<T>,
    derivative_bar: Vec<T>,
    multiplier: Vec<T>,
    state_weights: Vec<T::Real>,
    derivative_weights: Vec<T::Real>,
}
struct DaeWork<T: NumericScalar> {
    transpose: OdeStageBackend,
    primal: DaeStorage<T>,
    linear: LinearWork<T>,
    history_bar: Vec<T>,
    // Trapezoid cotangent of the recorded rate at the current reverse point.
    derivative_carry: Vec<T>,
    query: Vec<T>,
    rhs: Vec<T>,
    reverse: scioxide_linalg::Vector<T>,
    backward: crate::ivp::IvpStorage<T::Real, ContinuousWorkspace<T>, T>,
    reverse_attempts: usize,
}

impl<T: NumericScalar> DaeAdjointStorage<T> {
    fn fits(
        &self,
        tape: TapeShape,
        primal: &(DaeBdf, DaeOptions<T::Real>),
        transpose: &OdeStageBackend,
        provider: usize,
        objective: usize,
    ) -> bool {
        self.tape.fits(tape)
            && primal.0.fits(&self.work.primal)
            && self.work.transpose == *transpose
            && self.work.linear.products.provider.len() == provider
            && self.work.linear.products.objective.len() == objective
    }

    fn new(
        tape: TapeShape,
        primal: &(DaeBdf, DaeOptions<T::Real>),
        transpose: &OdeStageBackend,
        provider: usize,
        objective: usize,
    ) -> Result<Self, SciOxideError> {
        let n = tape.dimension();
        let p = tape.parameters();
        let filled = |len, quantity| scioxide_core::backend::try_filled(len, T::ZERO, quantity);
        let real = |len, quantity| scioxide_core::backend::try_filled(len, T::Real::ZERO, quantity);
        let mut backward = crate::ivp::IvpStorage::default();
        backward.reserve_driver(n + p, 0, 0)?;
        backward.method = ContinuousWorkspace {
            full: filled(n + p, scioxide_core::quantity!("reverse_full"))?,
            half: filled(n + p, scioxide_core::quantity!("reverse_half"))?,
            error: filled(n + p, scioxide_core::quantity!("reverse_error"))?,
        };
        Ok(Self {
            tape: Tape::new(tape)?,
            work: DaeWork {
                transpose: transpose.clone(),
                primal: primal.0.storage()?,
                linear: LinearWork {
                    solver: crate::dae::linear::DaeLinear::new(n, transpose, primal.0.backend())?,
                    products: Products::new(n, p, provider, objective)?,
                    derivative_bar: filled(n, scioxide_core::quantity!("derivative_bar"))?,
                    multiplier: filled(n, scioxide_core::quantity!("multiplier"))?,
                    state_weights: real(n, scioxide_core::quantity!("state_weights"))?,
                    derivative_weights: real(n, scioxide_core::quantity!("derivative_weights"))?,
                },
                history_bar: filled(MAX_ORDER * n, scioxide_core::quantity!("history_bar"))?,
                derivative_carry: filled(n, scioxide_core::quantity!("derivative_carry"))?,
                query: filled(2 * n, scioxide_core::quantity!("query"))?,
                rhs: filled(n, scioxide_core::quantity!("rhs"))?,
                reverse: scioxide_linalg::Vector::zeros(n + p)?,
                backward,
                reverse_attempts: 0,
            },
        })
    }
}

impl<T: NumericScalar> Default for DaeAdjointBuffers<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NumericScalar> core::fmt::Debug for DaeAdjointBuffers<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DaeAdjointBuffers")
            .field("prepared", &self.storage.is_some())
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> DaeAdjointBuffers<T> {
    // An empty workspace; allocates nothing.
    #[must_use]
    pub(super) const fn new() -> Self {
        Self {
            storage: None,
            shell: Shell::new(),
        }
    }

    fn fits(
        &self,
        tape: TapeShape,
        primal: &(DaeBdf, DaeOptions<T::Real>),
        transpose: &OdeStageBackend,
        provider: usize,
        objective: usize,
    ) -> bool {
        self.storage
            .as_ref()
            .is_some_and(|storage| storage.fits(tape, primal, transpose, provider, objective))
    }

    fn reserve(
        &mut self,
        tape: TapeShape,
        primal: &(DaeBdf, DaeOptions<T::Real>),
        transpose: &OdeStageBackend,
        provider: usize,
        objective: usize,
    ) -> Result<(), SciOxideError> {
        if !self.fits(tape, primal, transpose, provider, objective) {
            self.storage = None;
            self.storage = Some(DaeAdjointStorage::new(
                tape, primal, transpose, provider, objective,
            )?);
        }
        Ok(())
    }

    // Whether this workspace serves `requirements` without allocating.
    //
    // # Errors
    // This check is infallible today; the `Result` keeps the shared protocol.
    pub(super) fn check(
        &self,
        requirements: &DaeStorageRequirements<T>,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.fits(
                requirements.tape,
                &requirements.primal,
                &requirements.transpose,
                requirements.provider,
                requirements.objective,
            ) {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    // Prepare this workspace for `requirements`, replacing storage of another shape or
    // backend.
    //
    // # Errors
    // Returns backend preparation, allocation and size-overflow errors.
    pub(super) fn try_reserve(
        &mut self,
        requirements: &DaeStorageRequirements<T>,
    ) -> Result<(), SciOxideError> {
        self.reserve(
            requirements.tape,
            &requirements.primal,
            &requirements.transpose,
            requirements.provider,
            requirements.objective,
        )
    }

    // The last attempt: its epoch, outcome, counts and [`AdjointReport`] progress.
    pub(super) const fn last_attempt(&self) -> Option<&AttemptRecord<AdjointReport<T::Real>>> {
        self.shell.last_attempt()
    }
}

#[derive(Clone, Copy)]
enum Equation<S: RealScalar> {
    Step(S),
    Mixed,
    Continuous(S),
}
impl<T: NumericScalar> LinearWork<T> {
    #[allow(clippy::too_many_arguments)]
    fn equation<P: DaeAdjointSystem<T>>(
        &mut self,
        plan: &PreparedDaeAdjoint<T>,
        problem: &P,
        time: T::Real,
        point: &[T],
        equation: Equation<T::Real>,
        rhs: &[T],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = plan.partition.dimension();
        for (i, &differential) in plan.partition.variables().iter().enumerate() {
            let (a, b) = match equation {
                Equation::Step(inverse) => (T::Real::ONE, inverse),
                Equation::Mixed => {
                    if differential {
                        (T::Real::ZERO, T::Real::ONE)
                    } else {
                        (T::Real::ONE, T::Real::ZERO)
                    }
                }
                Equation::Continuous(h) => {
                    if differential {
                        (h, T::Real::ONE)
                    } else {
                        (T::Real::ONE, T::Real::ZERO)
                    }
                }
            };
            self.state_weights[i] = a;
            self.derivative_weights[i] = b;
        }
        let preconditioner = if matches!(plan.transpose_backend, OdeStageBackend::Krylov { .. }) {
            admit(context, EvaluationRole::Preconditioner)?;
            problem.adjoint_preconditioner(
                time,
                &point[..n],
                &point[n..2 * n],
                &self.state_weights,
                &self.derivative_weights,
            )?
        } else {
            None
        };
        let mut matrix = Transpose {
            problem,
            options: plan.primal.1.numerics(),
            partition: &plan.partition,
            time,
            state: &point[..n],
            derivative: &point[n..2 * n],
            state_weights: &self.state_weights,
            derivative_weights: &self.derivative_weights,
            original_pattern: match &plan.backend {
                OdeStageBackend::Sparse { pattern, .. } => Some(pattern.view()),
                _ => None,
            },
            permutation: &plan.permutation,
            parameter_scratch: RefCell::new(&mut self.products.parameters),
            provider_scratch: plan.provider_scratch,
            preconditioner,
        };
        self.solver.prepare(&mut matrix, None, context)?;
        let residual = self.solver.solve(
            &mut matrix,
            rhs,
            &mut self.multiplier,
            plan.linear_residual.relative(),
            context,
        )?;
        report.errors.linear_residual = Some(
            report
                .errors
                .linear_residual
                .unwrap_or(T::Real::ZERO)
                .max(residual),
        );
        self.pullback(problem, time, point, context)
    }
    fn pullback(
        &mut self,
        problem: &impl DaeAdjointSystem<T>,
        time: T::Real,
        point: &[T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.multiplier.len();
        admit(context, EvaluationRole::LinearOperator)?;
        self.products.state.fill(T::from_real(T::Real::NAN));
        self.derivative_bar.fill(T::from_real(T::Real::NAN));
        self.products.parameters.fill(T::Real::NAN);
        problem.pullback(
            time,
            &point[..n],
            &point[n..2 * n],
            &self.multiplier,
            &mut self.products.state,
            &mut self.derivative_bar,
            &mut self.products.parameters,
            &mut self.products.provider,
            context,
        )?;
        finite(
            &self.products.state,
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        finite(
            &self.derivative_bar,
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        finite(
            &self.products.parameters,
            scioxide_core::NonFiniteOrigin::Callback,
        )
    }
}

impl<T: NumericScalar> PreparedDaeAdjoint<T> {
    pub(super) fn matches(&self, shape: Shape<'_, T::Real>) -> bool {
        self.tape.dimension == shape.dimension
            && self.tape.parameters == shape.parameters
            && self.provider_scratch == shape.provider_scratch
            && self.objective_scratch == shape.objective_scratch
            && check_observations(&self.observations, shape.observations).is_ok()
    }

    // Prepare the shared variable-order residual BDF trajectory and its adjoint.
    //
    // Transposed linear equations accept the linear residual `linear_residual`.
    //
    // # Errors
    // Rejects incompatible dimensions, nonreplayable providers, nonfinite parameters,
    // invalid DAE or adjoint options, and checkpoint, linear or byte limits that cannot
    // be admitted, before any callback.
    pub(super) fn new(
        shape: Shape<'_, T::Real>,
        partition: DaePartition,
        backend: OdeStageBackend,
        dae_options: DaeNumericalOptions<T::Real>,
        options: Policies<T::Real>,
        linear_residual: RelativeTolerance<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            shape,
            partition,
            backend,
            (dae_options, linear_residual),
            options,
            Formula::Bdf {
                maximum_order: MAX_ORDER,
            },
        )
    }
    // Prepare single-step backward Euler for a pure differential residual.
    //
    // [`OdeAdjointResidual`] adapts normalized ODEs to this shared implicit kernel. The
    // order-one configuration uses the same BDF stepping and history machinery.
    //
    // # Errors
    // Returns the preparation errors of [`Self::new`].
    pub(super) fn backward_euler(
        shape: Shape<'_, T::Real>,
        backend: OdeStageBackend,
        dae_options: DaeNumericalOptions<T::Real>,
        options: Policies<T::Real>,
        linear_residual: RelativeTolerance<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::differential(
            shape,
            backend,
            (dae_options, linear_residual),
            options,
            Formula::Bdf { maximum_order: 1 },
        )
    }
    // Prepare the fixed-step implicit trapezoid (Crank-Nicolson) for a pure
    // differential residual and its exact discrete adjoint.
    //
    // Each step from `(t0, y0, v0)` solves `F(t0 + h, y1, v1) = 0` with
    // `v1 = 2 (y1 - y0) / h - v0` by full Newton on the shared residual kernel, so
    // checkpoint replay is bit-exact. The initial rate `v0` is the consistent rate of
    // the checked or projected initial pair. For a semi-explicit residual
    // `M ydot - f(t, y, p)` with constant regular `M`, such as [`OdeAdjointResidual`],
    // this is `M (y1 - y0) = h/2 (f0 + f1)`; a rate-dependent `F_ydot` gives a
    // different, still exactly differentiated, scheme. Running costs use the
    // trapezoidal rule `h/2 (L(t0, y0) + L(t1, y1))`.
    //
    // The reverse step carries the rate cotangent `c` with the state cotangent `lambda`:
    // with `A = F_y + (2/h) F_ydot` at the endpoint, it solves
    // `A^H mu = lambda + (2/h) c`, updates `c <- F_ydot^H mu - c`,
    // `lambda <- (2/h) c` and subtracts `F_p^T mu` from the parameter gradient; the
    // initial pullback eliminates the consistent initial rate. There is no primal or
    // quadrature error estimate: [`AdjointErrorReport::primal`] and
    // [`AdjointErrorReport::quadrature`] stay absent. Every step uses
    // [`Policies::initial_step`], shortened only at observation times, system
    // step limits and the final endpoint; a step whose Newton iteration fails ends the
    // solve with a rejection error.
    //

    //
    // # Errors
    // Rejects a [`AdjointMode::Continuous`] mode (`options.mode`) and an adaptive step
    // policy (`options.fixed_step`), and returns the preparation errors of
    // [`Self::new`].
    pub(super) fn trapezoid(
        shape: Shape<'_, T::Real>,
        backend: OdeStageBackend,
        dae_options: DaeNumericalOptions<T::Real>,
        options: Policies<T::Real>,
        linear_residual: RelativeTolerance<T::Real>,
    ) -> Result<Self, SciOxideError> {
        if options.mode != AdjointMode::Discrete {
            return Err(invalid(
                scioxide_core::argument!("options.mode"),
                "the trapezoid adjoint is discrete",
            ));
        }
        if !options.fixed_step {
            return Err(invalid(
                scioxide_core::argument!("options.fixed_step"),
                "the trapezoid adjoint requires fixed steps",
            ));
        }
        Self::differential(
            shape,
            backend,
            (dae_options, linear_residual),
            options,
            Formula::Trapezoid,
        )
    }
    // Shared preparation of the one-step formulas on an all-differential partition.
    fn differential(
        shape: Shape<'_, T::Real>,
        backend: OdeStageBackend,
        numerics: (DaeNumericalOptions<T::Real>, RelativeTolerance<T::Real>),
        options: Policies<T::Real>,
        formula: Formula,
    ) -> Result<Self, SciOxideError> {
        let minimum = storage_bound::<T>(
            &backend,
            shape.dimension,
            shape.parameters,
            checked_add(
                shape.provider_scratch,
                shape.objective_scratch,
                scioxide_core::quantity!("callback_scratch"),
            )?,
        )?;
        if minimum > options.max_workspace_bytes {
            return Err(invalid(
                scioxide_core::argument!("options.max_workspace_bytes"),
                "implicit adjoint workspace cannot fit the byte budget",
            ));
        }
        let flags = scioxide_core::backend::try_filled(
            shape.dimension,
            true,
            scioxide_core::quantity!("partition_flags"),
        )?;
        Self::prepare(
            shape,
            DaePartition::new(&flags, &flags)?,
            backend,
            numerics,
            options,
            formula,
        )
    }
    // `numerics` holds the DAE controls and the transposed linear-equation defect.
    fn prepare(
        shape: Shape<'_, T::Real>,
        partition: DaePartition,
        backend: OdeStageBackend,
        (dae_options, linear_residual): (DaeNumericalOptions<T::Real>, RelativeTolerance<T::Real>),
        options: Policies<T::Real>,
        formula: Formula,
    ) -> Result<Self, SciOxideError> {
        let observations = prepare_observations(shape.observations, options.mode)?;
        let n = partition.dimension();
        let p = shape.parameters;
        if n != shape.dimension {
            return Err(invalid(
                scioxide_core::argument!("partition"),
                "partition dimension differs from the plan",
            ));
        }
        let differential = partition.variables().iter().filter(|&&x| x).count();
        // A restart holds the accepted pair and, for BDF, its history. A frame holds
        // the start state, the endpoint pair, the restart history and one running
        // cost per quadrature node.
        let (pairs, nodes) = match formula {
            Formula::Bdf { .. } => (MAX_ORDER + 2, MAX_ORDER + 1),
            Formula::Trapezoid => (2, 2),
        };
        let restart = checked_mul(n, pairs, scioxide_core::quantity!("restart_state"))?;
        options.validate(n, p, restart, false)?;
        let base = storage_bound::<T>(
            &backend,
            n,
            p,
            checked_add(
                shape.provider_scratch,
                shape.objective_scratch,
                scioxide_core::quantity!("callback_scratch"),
            )?,
        )?;
        let frame = checked_add(
            checked_mul(n, pairs + 1, scioxide_core::quantity!("checkpoint_frame"))?,
            nodes,
            scioxide_core::quantity!("checkpoint_frame"),
        )?;
        let tape = TapePlan::new::<T, BdfStep<T::Real>>(
            n,
            p,
            frame,
            restart,
            base,
            options.limits(),
            &options.checkpoint_policy,
            !observations.is_empty(),
        )?;
        let (transpose_backend, permutation) =
            prepare_backend(&backend, n, shape.provider_scratch)?;
        let projection_options = dae_options
            .clone()
            .with_initialization(DaeInitialization::Project);
        let primal = (
            DaeBdf::new(partition.clone(), backend.clone())?,
            DaeOptions::new().with_numerics(dae_options),
        );
        let projection = (
            DaeBdf::new(partition.clone(), backend.clone())?,
            DaeOptions::new().with_numerics(projection_options),
        );
        Ok(Self {
            tape,
            observations,
            options,
            linear_residual,
            primal,
            projection,
            partition,
            backend,
            transpose_backend,
            permutation,
            provider_scratch: shape.provider_scratch,
            objective_scratch: shape.objective_scratch,
            differential,
            formula,
        })
    }

    // The validated adjoint options.

    // The relative defect accepted for transposed linear equations.

    // Variable and equation roles of the residual.

    // Storage needed by every solve with this plan.
    //
    // # Errors
    // Returns allocation errors of copying the stage backends.
    pub(super) fn workspace_requirements(
        &self,
    ) -> Result<DaeStorageRequirements<T>, SciOxideError> {
        Ok(DaeStorageRequirements {
            tape: self.tape.shape(),
            primal: self.primal.clone(),
            transpose: self.transpose_backend.clone(),
            provider: self.provider_scratch,
            objective: self.objective_scratch,
            differential: self.differential,
            bytes: self.tape.bytes,
        })
    }

    fn reserve(&self, workspace: &mut DaeAdjointBuffers<T>) -> Result<(), SciOxideError> {
        workspace.reserve(
            self.tape.shape(),
            &self.primal,
            &self.transpose_backend,
            self.provider_scratch,
            self.objective_scratch,
        )
    }

    // A workspace with every primal, reverse, projection and checkpoint buffer prepared;
    // solves never grow it.
    //
    // # Errors
    // Returns size, allocation or backend preparation failures.
    #[cfg(test)]
    pub(super) fn workspace(&self) -> Result<DaeAdjointBuffers<T>, SciOxideError> {
        let mut workspace = DaeAdjointBuffers::new();
        self.reserve(&mut workspace)?;
        Ok(workspace)
    }

    // Shape, capability, interval and finiteness checks; grows an undersized workspace.
    #[allow(clippy::too_many_arguments)]
    fn preflight(
        &self,
        problem: &impl DaeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        derivative: &[T],
        interval: (T::Real, T::Real),
        initial_gradient: &[T],
        parameter_gradient: &[T::Real],
        workspace: &mut DaeAdjointBuffers<T>,
    ) -> Result<(), SciOxideError> {
        check_observations(&self.observations, objective.observation_times())?;
        let n = self.tape.dimension;
        if problem.dimension() != n
            || initial.len() != n
            || derivative.len() != n
            || initial_gradient.len() != self.differential
            || parameter_gradient.len() != self.tape.parameters
            || problem.parameters().len() != self.tape.parameters
            || problem.pullback_workspace_len() > self.provider_scratch
            || objective.workspace_len() > self.objective_scratch
            || !problem.is_replayable()
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "DAE adjoint plan, providers and gradient outputs must match",
            ));
        }
        if !interval.0.is_finite() || !interval.1.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("interval"),
                "finite DAE adjoint interval required",
            ));
        }
        finite(initial, scioxide_core::NonFiniteOrigin::Input)?;
        finite(derivative, scioxide_core::NonFiniteOrigin::Input)?;
        finite(
            problem.parameters(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        self.reserve(workspace)
    }

    // Compute initial-manifold and real-parameter gradients of the objective.
    //
    // The initial pair is checked or projected according to the primal
    // [`DaeOptions`]. Each call is one epoch of `scope` (a root epoch for an
    // [`ExecutionContext`], a child epoch for a `&mut SolveContext`) and records a fresh
    // tape. Both gradient outputs are written only on success and remain unchanged on
    // every error; diagnostics of a failed attempt, classified as in
    // [`AdjointReport::termination`], are recorded in
    // [`DaeAdjointBuffers::last_attempt`]. A preflight rejection runs no callback; an
    // undersized workspace is grown first. With a prepared workspace and nonallocating
    // callbacks, the numerical processing allocates nothing.
    //
    // # Errors
    // Returns shape, capability, interval and nonfinite-input errors at preflight, and
    // consistency, provider, rank, replay, numerical, budget and cancellation errors
    // during the solve.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_into<'scope, 'capability>(
        &self,
        problem: &impl DaeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        derivative: &[T],
        interval: (T::Real, T::Real),
        initial_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
        workspace: &mut DaeAdjointBuffers<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdjointReport<T::Real>, SciOxideError> {
        let scope = scope.into();
        if let Err(error) = self.preflight(
            problem,
            objective,
            initial,
            derivative,
            interval,
            initial_gradient,
            parameter_gradient,
            workspace,
        ) {
            return workspace.shell.reject(error);
        }
        let DaeAdjointBuffers { storage, shell } = workspace;
        let Some(storage) = storage.as_mut() else {
            return shell.reject(invalid(
                scioxide_core::argument!("workspace"),
                "DAE adjoint workspace storage is not prepared",
            ));
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            match self.formula {
                Formula::Bdf { .. } => scioxide_core::algorithm_id!("ode", "index1-adjoint", 1),
                Formula::Trapezoid => scioxide_core::algorithm_id!("ode", "trapezoid-adjoint", 1),
            },
            self.tape.dimension as u128,
            self.options.mode as u128,
        );
        let report = execute(
            shell,
            scope,
            identity,
            self.options.budgets(),
            AdjointReport::initial(interval.0, self.tape.bytes),
            |report, cx| {
                self.run(
                    problem, objective, initial, derivative, interval, storage, report, cx,
                )
            },
        )?;
        for (out, &value) in initial_gradient.iter_mut().zip(
            self.partition
                .variables()
                .iter()
                .zip(&storage.tape.lambda)
                .filter_map(|(&differential, value)| differential.then_some(value)),
        ) {
            *out = value;
        }
        parameter_gradient.copy_from_slice(&storage.tape.gradient);
        Ok(report)
    }

    // Consistency processing, forward recording and reverse sweep of one epoch.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        problem: &impl DaeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        derivative: &[T],
        interval: (T::Real, T::Real),
        storage: &mut DaeAdjointStorage<T>,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = self.tape.dimension;
        let DaeAdjointStorage { tape, work } = storage;
        tape.state[..n].copy_from_slice(initial);
        tape.state[n..2 * n].copy_from_slice(derivative);
        tape.bind_parameters(problem.parameters())?;
        consistency(
            &mut work.primal,
            &self.primal,
            problem,
            &mut tape.state[..2 * n],
            interval.0,
            report,
            context,
        )?;
        report.trust = scioxide_core::PartialResultTrust::ResidualChecked;
        work.reverse_attempts = 0;
        work.history_bar.fill(T::ZERO);
        work.derivative_carry.fill(T::ZERO);
        // BDF starts with every history slot at the initial state.
        for j in 2..self.tape.restart_len / n {
            tape.state.copy_within(..n, j * n);
        }
        let mut kernel = DaeKernel {
            plan: self,
            problem,
            objective,
            work,
        };
        let terminal_event = tape.run(
            &self.tape,
            &self.options,
            &mut kernel,
            interval.0,
            interval.1,
            report,
            context,
        )?;
        check_observations(&self.observations, objective.observation_times())?;
        context.execution().check()?;
        Ok(terminal_event)
    }

    // Compute both gradients with an internal workspace under sequential native
    // execution (portable scalar dispatch without `std`).
    //
    // # Errors
    // Returns the [`Self::solve_into`] error together with the last attempt's progress
    // when one was recorded.
}

// Check or project a residual pair in a child epoch of `context`. Inconsistent data are
// invalid input; projection exhaustion is an adjoint iteration limit.
fn consistency<T: NumericScalar>(
    solver: &mut DaeStorage<T>,
    plan: &(DaeBdf, DaeOptions<T::Real>),
    problem: &impl DaeAdjointSystem<T>,
    point: &mut [T],
    time: T::Real,
    report: &mut AdjointReport<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = problem.dimension();
    let (state, derivative) = point.split_at_mut(n);
    let mut epoch = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("ode", "adjoint-consistency", 1),
        n as u128,
        0,
    );
    SolveScope::from(&mut *context).run(&mut epoch, identity, SolveBudgets::unbounded(), |child| {
        let mut projection = DaeReport::initial(time);
        solver
            .project(
                (&plan.0, &plan.1),
                problem,
                state,
                derivative,
                time,
                child,
                &mut projection,
            )?
            .map_err(|failure| match failure {
                Unaccepted::Inconsistent => SciOxideError::invalid_input(
                    scioxide_core::argument!("problem"),
                    InvalidInputReason::Inconsistent,
                    "adjoint DAE initial data are inconsistent",
                ),
                Unaccepted::Projection(..) => {
                    stop_error(Stop::Budget(BudgetKind::Iteration), report)
                }
            })
    })
}

struct DaeKernel<'a, T: NumericScalar, P, O> {
    plan: &'a PreparedDaeAdjoint<T>,
    problem: &'a P,
    objective: &'a O,
    work: &'a mut DaeWork<T>,
}

impl<T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>> Kernel<T>
    for DaeKernel<'_, T, P, O>
{
    type Metadata = BdfStep<T::Real>;
    fn observations(&self) -> &[T::Real] {
        &self.plan.observations
    }
    fn sample(
        &mut self,
        time: T::Real,
        state: &[T],
        packed: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.work.linear.products.observations(
            self.objective,
            &self.plan.observations,
            time,
            state,
            self.problem.parameters(),
            packed,
            context,
        )
    }
    fn parameters(&self) -> &[T::Real] {
        self.problem.parameters()
    }
    fn record(
        &mut self,
        tape: &mut Tape<T, BdfStep<T::Real>>,
        start: T::Real,
        end: T::Real,
        cost: &mut T::Real,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        let mut progress = DaeReport::initial(start);
        let mut recorder = Forward {
            plan: self.plan,
            problem: self.problem,
            objective: self.objective,
            tape,
            linear: &mut self.work.linear,
            end,
            cost,
            report,
            quadrature: None,
            running_cost: T::Real::ZERO,
        };
        let rejected_before = context.counts().own().rejected();
        let result = self.work.primal.run_steps(
            (&self.plan.primal.0, &self.plan.primal.1),
            self.problem,
            start,
            end,
            self.plan.options.initial_step,
            context,
            &mut |_, _, _| Ok(()),
            &mut progress,
            &mut recorder,
        );
        recorder.report.rejected_steps += context.counts().own().rejected() - rejected_before;
        if progress.has_accepted_state() {
            recorder.report.trust = progress.trust();
        }
        if let Some(error) = progress.last_step_error_estimate() {
            recorder.report.last_step_error_estimate = Some(error);
        }
        result?;
        dae_termination(progress.termination(), recorder.report)?;
        Ok(false)
    }
    fn replay(
        &mut self,
        _index: usize,
        step: Step<T::Real>,
        initial: &[T],
        candidate: &mut [T],
        frame: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        match self.plan.formula {
            Formula::Bdf { .. } => self.work.primal.replay_bdf(
                (&self.plan.primal.0, &self.plan.primal.1),
                self.problem,
                step.time,
                step.width,
                initial,
                step.metadata,
                context,
            )?,
            Formula::Trapezoid => self.work.primal.replay_trapezoid(
                (&self.plan.primal.0, &self.plan.primal.1),
                self.problem,
                step.time,
                step.width,
                initial,
                context,
            )?,
        }
        prepare_frame(
            self.plan,
            self.problem,
            self.objective,
            &mut self.work.linear.products,
            &self.work.primal,
            step.time,
            step.width,
            step.metadata,
            initial,
            candidate,
            frame,
            false,
            context,
        )?;
        Ok(())
    }
    fn terminal(
        &mut self,
        time: T::Real,
        state: &[T],
        _: bool,
        _: Option<Step<T::Real>>,
        _: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        let n = lambda.len();
        let value = self
            .work
            .linear
            .products
            .objective(
                self.objective,
                AdjointObjectivePart::Terminal,
                time,
                &state[..n],
                self.problem.parameters(),
                context,
            )?
            .value;
        lambda.copy_from_slice(&self.work.linear.products.state);
        gradient.copy_from_slice(&self.work.linear.products.parameters);
        if self.plan.options.mode == AdjointMode::Continuous {
            self.reduce(time, state, lambda, gradient, report, context)?;
        }
        Ok(value)
    }
    fn reverse(
        &mut self,
        step: Step<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = lambda.len();
        let geometry = step.metadata;
        if self.plan.options.mode == AdjointMode::Continuous {
            return self.continuous(
                step.time, step.width, geometry, frame, lambda, gradient, report, context,
            );
        }
        if self.plan.formula == Formula::Trapezoid {
            return self.trapezoid(step, frame, lambda, gradient, report, context);
        }
        let q = geometry.order;
        let quadrature = weights(&geometry.nodes[..q], false)?;
        for (j, &weight) in quadrature.iter().enumerate().take(q) {
            let offset = match j {
                0 => n,
                1 => 0,
                _ => (j + 1) * n,
            };
            self.work.linear.products.objective(
                self.objective,
                AdjointObjectivePart::Running,
                step.time + step.width * geometry.nodes[j],
                &frame[offset..offset + n],
                self.problem.parameters(),
                context,
            )?;
            let weight = step.width * weight;
            for (i, &value) in self.work.linear.products.state.iter().enumerate() {
                if j == 0 {
                    lambda[i] = lambda[i] + T::from_real(weight) * value;
                } else {
                    self.work.history_bar[(j - 1) * n + i] =
                        self.work.history_bar[(j - 1) * n + i] + T::from_real(weight) * value;
                }
            }
            for (out, &value) in gradient
                .iter_mut()
                .zip(&self.work.linear.products.parameters)
            {
                *out += weight * value;
            }
        }
        self.work.rhs.copy_from_slice(lambda);
        self.work.linear.equation(
            self.plan,
            self.problem,
            step.endpoint,
            &frame[n..3 * n],
            Equation::Step(geometry.alpha[0] / step.width),
            &self.work.rhs,
            report,
            context,
        )?;
        for j in 1..=q {
            for (i, &value) in self.work.linear.derivative_bar.iter().enumerate() {
                self.work.history_bar[(j - 1) * n + i] = crate::safety::state_increment(
                    self.work.history_bar[(j - 1) * n + i],
                    -geometry.alpha[j] / step.width,
                    value,
                )?;
            }
        }
        for (out, &value) in gradient
            .iter_mut()
            .zip(&self.work.linear.products.parameters)
        {
            *out -= value;
        }
        lambda.copy_from_slice(&self.work.history_bar[..n]);
        self.work.history_bar.copy_within(n.., 0);
        self.work.history_bar[(MAX_ORDER - 1) * n..].fill(T::ZERO);
        finite(lambda, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(gradient, scioxide_core::NonFiniteOrigin::Callback)
    }
    fn initial(
        &mut self,
        time: T::Real,
        state: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.plan.options.mode == AdjointMode::Discrete {
            self.reduce(time, state, lambda, gradient, report, context)?;
        }
        Ok(())
    }
}

impl<T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>> DaeKernel<'_, T, P, O> {
    // Pull the cotangents of the consistent initial unknowns, the differential rates
    // (`derivative_carry`, zero for BDF) and the algebraic states (`lambda`), back to
    // the free differential states and the parameters through `F(t0, y0, v0) = 0`.
    fn reduce(
        &mut self,
        time: T::Real,
        state: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, &differential) in self.plan.partition.variables().iter().enumerate() {
            self.work.rhs[i] = if differential {
                self.work.derivative_carry[i]
            } else {
                lambda[i]
            };
        }
        self.work.linear.equation(
            self.plan,
            self.problem,
            time,
            state,
            Equation::Mixed,
            &self.work.rhs,
            report,
            context,
        )?;
        for (i, out) in lambda.iter_mut().enumerate() {
            *out = if self.plan.partition.variables()[i] {
                *out - self.work.linear.products.state[i]
            } else {
                T::ZERO
            };
        }
        for (out, &v) in gradient
            .iter_mut()
            .zip(&self.work.linear.products.parameters)
        {
            *out -= v;
        }
        Ok(())
    }
    // Discrete reverse of one implicit-trapezoid step `t0 -> t1 = t0 + h`.
    //
    // The step solves `F(t1, y1, v1) = 0` with `v1 = (2/h)(y1 - y0) - v0` and adds
    // `h/2 (L(t0, y0) + L(t1, y1))`. From the cotangents `lambda` of `y1` and `c` of
    // `v1`, with `A = F_y + (2/h) F_ydot` at `(t1, y1, v1)`, it solves
    // `A^H mu = lambda + h/2 L_y(t1) + (2/h) c` and returns the start cotangents
    // `c0 = F_ydot^H mu - c` and `lambda0 = (2/h) c0 + h/2 L_y(t0)`; the parameter
    // gradient gains `h/2 (L_p(t0) + L_p(t1)) - F_p^T mu`.
    fn trapezoid(
        &mut self,
        step: Step<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = lambda.len();
        let end = step.time + step.width;
        let half = step.width * T::Real::HALF;
        let inverse = T::Real::TWO / step.width;
        self.running(end, &frame[n..2 * n], half, lambda, gradient, context)?;
        for (rhs, (&state, &rate)) in self
            .work
            .rhs
            .iter_mut()
            .zip(lambda.iter().zip(&self.work.derivative_carry))
        {
            *rhs = crate::safety::state_increment(state, inverse, rate)?;
        }
        self.work.linear.equation(
            self.plan,
            self.problem,
            end,
            &frame[n..3 * n],
            Equation::Step(inverse),
            &self.work.rhs,
            report,
            context,
        )?;
        for (out, &value) in gradient
            .iter_mut()
            .zip(&self.work.linear.products.parameters)
        {
            *out -= value;
        }
        for ((carry, out), &value) in self
            .work
            .derivative_carry
            .iter_mut()
            .zip(lambda.iter_mut())
            .zip(&self.work.linear.derivative_bar)
        {
            *carry = crate::safety::state_difference(value, *carry)?;
            *out = crate::safety::state_increment(T::ZERO, inverse, *carry)?;
        }
        self.running(step.time, &frame[..n], half, lambda, gradient, context)?;
        finite(
            &self.work.derivative_carry,
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        finite(lambda, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(gradient, scioxide_core::NonFiniteOrigin::Callback)
    }
    // Add `weight` times the running-cost gradients at `(time, state)`.
    fn running(
        &mut self,
        time: T::Real,
        state: &[T],
        weight: T::Real,
        lambda: &mut [T],
        gradient: &mut [T::Real],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let products = &mut self.work.linear.products;
        products.objective(
            self.objective,
            AdjointObjectivePart::Running,
            time,
            state,
            self.problem.parameters(),
            context,
        )?;
        for (out, &value) in lambda.iter_mut().zip(&products.state) {
            *out = *out + T::from_real(weight) * value;
        }
        for (out, &value) in gradient.iter_mut().zip(&products.parameters) {
            *out += weight * value;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn continuous(
        &mut self,
        start: T::Real,
        width: T::Real,
        geometry: BdfStep<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = lambda.len();
        self.work.reverse.as_mut_slice()[..n].copy_from_slice(lambda);
        for (out, &value) in self.work.reverse.as_mut_slice()[n..]
            .iter_mut()
            .zip(gradient.iter())
        {
            *out = T::from_real(value);
        }
        let shared_report = RefCell::new(report);
        let trial = ContinuousTrial {
            plan: self.plan,
            problem: self.problem,
            objective: self.objective,
            start,
            width,
            geometry,
            frame,
            report: &shared_report,
            work: RefCell::new(ContinuousWork {
                primal: &mut self.work.primal,
                linear: &mut self.work.linear,
                query: &mut self.work.query,
                rhs: &mut self.work.rhs,
            }),
        };
        let mut recorder = super::reverse::Reverse {
            options: &self.plan.options,
            dimension: n,
            attempts: &mut self.work.reverse_attempts,
            limit: self.plan.tape.limits.max_reverse_steps,
            restart: true,
            report: &shared_report,
            adjoint: T::Real::ZERO,
            quadrature: T::Real::ZERO,
        };
        // The drive runs in the segment offset `s = t − start`, from `width` to zero. Its
        // first trial and any final step land exactly on zero, so no remainder below the
        // time resolution of `start` is left.
        let options = crate::OdeOptions::new().with_initial_step(width.abs());
        let run = crate::ivp::TrialDrive {
            info: crate::stepper::MethodInfo {
                name: "DAE continuous adjoint",
                algorithm: scioxide_core::algorithm_id!("ode", "dae-continuous-adjoint", 1),
                order: 1,
                initial_step_order: 1,
                controller: crate::stepper::DefaultController::Integral,
                dense_output: false,
                events: false,
                adaptive: true,
                mass_matrices: false,
                injectable: false,
            },
            options: &options,
            tolerance: &self.plan.options.adjoint_error,
            events: &[],
            delay: None,
            provider: crate::stiff::stage::Provider::Native,
            end: T::Real::ZERO,
        };
        let mut offset = width;
        let result = crate::ivp::drive_trials(
            &run,
            &trial,
            &mut self.work.backward,
            &mut offset,
            &mut self.work.reverse,
            &mut (),
            &mut recorder,
            context,
        )?;
        ivp_finish(result.finish, &mut shared_report.borrow_mut())?;
        lambda.copy_from_slice(&self.work.reverse.as_slice()[..n]);
        for (out, value) in gradient.iter_mut().zip(&self.work.reverse.as_slice()[n..]) {
            *out = value.real_component();
        }
        Ok(())
    }
}

// Prepared full/half-step estimates; the accepted state belongs to the driver.
struct ContinuousWorkspace<T> {
    full: Vec<T>,
    half: Vec<T>,
    error: Vec<T>,
}
impl<T> Default for ContinuousWorkspace<T> {
    fn default() -> Self {
        Self {
            full: Vec::new(),
            half: Vec::new(),
            error: Vec::new(),
        }
    }
}
struct ContinuousWork<'a, T: NumericScalar> {
    primal: &'a mut DaeStorage<T>,
    linear: &'a mut LinearWork<T>,
    query: &'a mut [T],
    rhs: &'a mut [T],
}
struct ContinuousTrial<'a, 'r, T: NumericScalar, P, O> {
    plan: &'a PreparedDaeAdjoint<T>,
    problem: &'a P,
    objective: &'a O,
    start: T::Real,
    width: T::Real,
    geometry: BdfStep<T::Real>,
    frame: &'a [T],
    report: &'a RefCell<&'r mut AdjointReport<T::Real>>,
    work: RefCell<ContinuousWork<'a, T>>,
}
impl<T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>>
    crate::ivp::TrialKernel<T::Real, T> for ContinuousTrial<'_, '_, T, P, O>
{
    const ADAPTIVE: Option<bool> = Some(true);
    type Workspace = ContinuousWorkspace<T>;
    fn begin(&self, _: usize, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn initial_step(
        &self,
        _: &mut crate::ivp::IvpStorage<T::Real, Self::Workspace, T>,
        _: T::Real,
        _: &scioxide_linalg::Vector<T>,
        _: T::Real,
        _: crate::stepper::StepRequest<'_, T::Real, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        Ok(self.width.abs())
    }
    fn step_limit(
        &self,
        _: T::Real,
        _: &[T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<T::Real>, SciOxideError> {
        Ok(None)
    }
    fn admits(
        &self,
        _: T::Real,
        _: &[T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok(true)
    }
    // `offset` and the step are segment offsets `t − start` (see `continuous`).
    fn step<'w>(
        &self,
        offset: T::Real,
        candidate: &mut scioxide_linalg::Vector<T>,
        h: T::Real,
        request: crate::stepper::StepRequest<'_, T::Real, T>,
        workspace: &'w mut Self::Workspace,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<crate::result::StepResult<'w, T::Real, T>, SciOxideError> {
        let end = request.endpoint(offset, h)?;
        let middle = offset * T::Real::HALF + end * T::Real::HALF;
        if middle == offset || middle == end {
            return Err(convergence(
                scioxide_core::ConvergenceReason::StepSizeLimit,
                "DAE continuous reverse time cannot advance",
            ));
        }
        let input = request
            .base
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("step"),
                    "DAE reverse trial requires an accepted base",
                )
            })?
            .1
            .as_slice();
        let mut work = self.work.borrow_mut();
        let ContinuousWork {
            primal,
            linear,
            query,
            rhs,
        } = &mut *work;
        let mut report = self.report.borrow_mut();
        let mut advance = |at, width, input: &[T], output: &mut [T]| {
            // Projection and transpose solves retain their own failure classification;
            // they are nested providers, not failures of this driver's trial kernel.
            request.trace.provider(continuous_step(
                self.plan,
                self.problem,
                self.objective,
                primal,
                linear,
                query,
                rhs,
                self.start,
                self.width,
                self.geometry,
                self.frame,
                at,
                width,
                input,
                output,
                &mut report,
                context,
            ))
        };
        advance(end, -h, input, &mut workspace.full)?;
        advance(middle, offset - middle, input, &mut workspace.half)?;
        advance(end, middle - end, &workspace.half, candidate.as_mut_slice())?;
        // The difference of two half steps and one full step estimates the half-step
        // error; local Richardson extrapolation `2 y_half − y_full` advances the
        // second-order result.
        for ((error, half), &full) in workspace
            .error
            .iter_mut()
            .zip(candidate.as_mut_slice())
            .zip(&workspace.full)
        {
            *error = crate::safety::state_difference(*half, full)?;
            *half = crate::safety::state_increment(*half, T::Real::ONE, *error)?;
        }
        let mut result =
            crate::result::StepResult::from_checked_parts(Some(&workspace.error), None);
        result.error_order = Some(2);
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)]
fn continuous_step<T: NumericScalar>(
    plan: &PreparedDaeAdjoint<T>,
    problem: &impl DaeAdjointSystem<T>,
    objective: &impl AdjointObjective<T>,
    primal: &mut DaeStorage<T>,
    linear: &mut LinearWork<T>,
    point: &mut [T],
    rhs: &mut [T],
    start: T::Real,
    width: T::Real,
    geometry: BdfStep<T::Real>,
    frame: &[T],
    offset: T::Real,
    h: T::Real,
    input: &[T],
    output: &mut [T],
    report: &mut AdjointReport<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = plan.tape.dimension;
    // Offsets lie between zero and `width`, so `theta` lies in [0, 1] and is exact at
    // both recorded endpoints, as is `start + width`.
    let theta = offset / width;
    let time = start + offset;
    for (i, out) in point[..n].iter_mut().enumerate() {
        let polynomial = Polynomial::new(&geometry.nodes[..=geometry.order], |j| {
            let offset = match j {
                0 => n,
                1 => 0,
                _ => (j + 1) * n,
            };
            frame[offset + i]
        })?;
        *out = polynomial.value(theta)?;
        rhs[i] = crate::safety::StageValue::from_value(polynomial.derivative(theta)?)?
            .div(crate::safety::StageValue::from_value(width)?)?
            .round()?;
    }
    point[n..].copy_from_slice(&frame[2 * n..3 * n]);
    consistency(
        primal,
        &plan.projection,
        problem,
        point,
        time,
        report,
        context,
    )?;
    linear.products.objective(
        objective,
        AdjointObjectivePart::Running,
        time,
        &point[..n],
        problem.parameters(),
        context,
    )?;
    let mut defect = report.errors.interpolation_defect.unwrap_or(T::Real::ZERO);
    for (i, &differential) in plan.partition.variables().iter().enumerate() {
        if differential {
            let difference = (rhs[i] - point[n + i]).abs();
            if !difference.is_finite() {
                return Err(invalid(
                    scioxide_core::argument!("problem"),
                    "nonfinite DAE reconstruction defect",
                ));
            }
            defect = defect.max(difference);
        }
    }
    report.errors.interpolation_defect = Some(defect);
    for (i, out) in rhs.iter_mut().enumerate() {
        *out = if plan.partition.variables()[i] {
            input[i] + T::from_real(h) * linear.products.state[i]
        } else {
            linear.products.state[i]
        };
    }
    for (out, (&value, &cost)) in output[n..]
        .iter_mut()
        .zip(input[n..].iter().zip(&linear.products.parameters))
    {
        *out = value + T::from_real(h * cost);
    }
    linear.equation(
        plan,
        problem,
        time,
        point,
        Equation::Continuous(h),
        rhs,
        report,
        context,
    )?;
    for (i, out) in output[..n].iter_mut().enumerate() {
        *out = if plan.partition.variables()[i] {
            linear.derivative_bar[i]
        } else {
            T::ZERO
        };
    }
    for (out, &value) in output[n..].iter_mut().zip(&linear.products.parameters) {
        *out = *out - T::from_real(h * value);
    }
    finite(output, scioxide_core::NonFiniteOrigin::Callback)
}

struct Forward<'a, T: NumericScalar, P, O> {
    plan: &'a PreparedDaeAdjoint<T>,
    problem: &'a P,
    objective: &'a O,
    tape: &'a mut Tape<T, BdfStep<T::Real>>,
    linear: &'a mut LinearWork<T>,
    end: T::Real,
    cost: &'a mut T::Real,
    report: &'a mut AdjointReport<T::Real>,
    quadrature: Option<T::Real>,
    running_cost: T::Real,
}
impl<T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>> crate::dae::Recorder<T>
    for Forward<'_, T, P, O>
{
    fn boundary(&mut self, time: T::Real) -> Result<Option<T::Real>, SciOxideError> {
        super::checkpoint::check_parameters(&self.tape.parameters, self.problem.parameters())?;
        self.tape.boundary(
            &self.plan.tape,
            &self.plan.observations,
            time,
            self.end,
            self.report,
        )
    }
    fn reuse_between_steps(&self) -> bool {
        false
    }
    fn fixed_step(&self) -> bool {
        self.plan.options.fixed_step
    }
    fn formula(&self) -> Formula {
        self.plan.formula
    }
    fn trial(
        &mut self,
        primal: &DaeStorage<T>,
        time: T::Real,
        width: T::Real,
        trial: crate::dae::bdf::Trial<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        if !trial.converged {
            return Ok(T::Real::ZERO);
        }
        let geometry = trial.geometry;
        let initial = &self.tape.state;
        let candidate = &mut self.tape.candidate;
        let frame = &mut self.tape.frame[..self.plan.tape.numerical_frame_len];
        let (cost, qerror) = prepare_frame(
            self.plan,
            self.problem,
            self.objective,
            &mut self.linear.products,
            primal,
            time,
            width,
            geometry,
            initial,
            candidate,
            frame,
            true,
            context,
        )?;
        self.quadrature = qerror;
        self.running_cost = cost;
        Ok(qerror.unwrap_or(T::Real::ZERO))
    }
    fn accepted(
        &mut self,
        _: &DaeStorage<T>,
        start: T::Real,
        width: T::Real,
        end: T::Real,
        trial: crate::dae::bdf::Trial<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.linear.products.observations(
            self.objective,
            &self.plan.observations,
            end,
            &self.tape.candidate,
            self.problem.parameters(),
            &mut self.tape.frame[self.plan.tape.numerical_frame_len..],
            context,
        )?;
        super::checkpoint::check_parameters(&self.tape.parameters, self.problem.parameters())?;
        self.tape.push(
            &self.plan.tape,
            start,
            width,
            end,
            Trial {
                error: match self.plan.formula {
                    Formula::Bdf { .. } => Some(trial.errors.0.max(trial.errors.1)),
                    Formula::Trapezoid => None,
                },
                quadrature_error: self.quadrature,
                cost: self.running_cost,
                fraction: T::Real::ONE,
                metadata: trial.geometry,
            },
            self.cost,
            self.report,
        )
    }
}

// Classify the unsuccessful termination of a nested DAE recording.
//
// Shared stops record their classification; rejection exhaustion and events are
// unclassified failures.
fn dae_termination<S: RealScalar>(
    termination: Option<DaeTermination>,
    report: &mut AdjointReport<S>,
) -> Result<(), SciOxideError> {
    use scioxide_core::ConvergenceReason as Reason;
    match termination {
        None | Some(DaeTermination::ReachedEnd) => Ok(()),
        Some(DaeTermination::BudgetLimit(kind)) => Err(stop_error(Stop::Budget(kind), report)),
        Some(DaeTermination::StepSizeLimit) => Err(stop_error(Stop::StepSize, report)),
        Some(DaeTermination::ExecutionStopped(reason)) => {
            Err(stop_error(Stop::Execution(reason), report))
        }
        Some(DaeTermination::RejectionLimit | DaeTermination::TerminalEvent) => Err(convergence(
            Reason::NoSuccessfulCandidate,
            "unsuccessful nested DAE integration",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_frame<T: NumericScalar>(
    plan: &PreparedDaeAdjoint<T>,
    problem: &impl DaeAdjointSystem<T>,
    objective: &impl AdjointObjective<T>,
    products: &mut Products<T>,
    primal: &DaeStorage<T>,
    time: T::Real,
    width: T::Real,
    geometry: BdfStep<T::Real>,
    initial: &[T],
    candidate: &mut [T],
    frame: &mut [T],
    estimate: bool,
    context: &mut SolveContext<'_, '_>,
) -> Result<(T::Real, Option<T::Real>), SciOxideError> {
    let n = plan.tape.dimension;
    frame.fill(T::ZERO);
    let cost;
    let mut qerror = None;
    {
        // Restart: the accepted pair, then any BDF history. Frame: the start state,
        // the endpoint pair, the start history and the running cost of each node.
        let restart = plan.tape.restart_len;
        let (y, d) = primal.candidate();
        candidate[..n].copy_from_slice(y);
        candidate[n..2 * n].copy_from_slice(d);
        if restart > 2 * n {
            candidate[2 * n..3 * n].copy_from_slice(&initial[..n]);
            candidate[3 * n..].copy_from_slice(&initial[2 * n..restart - n]);
        }
        frame[..n].copy_from_slice(&initial[..n]);
        frame[n..3 * n].copy_from_slice(&candidate[..2 * n]);
        frame[3 * n..n + restart].copy_from_slice(&initial[2 * n..]);
        let q = geometry.order;
        let mut values = [T::Real::ZERO; 8];
        for (j, value) in values.iter_mut().enumerate().take(q + 1) {
            let offset = match j {
                0 => n,
                1 => 0,
                _ => (j + 1) * n,
            };
            *value = products
                .objective(
                    objective,
                    AdjointObjectivePart::Running,
                    time + width * geometry.nodes[j],
                    &frame[offset..offset + n],
                    problem.parameters(),
                    context,
                )?
                .value;
            frame[n + restart + j] = T::from_real(*value);
        }
        if plan.formula == Formula::Trapezoid {
            // The trapezoidal rule over the endpoint (node one) and start (node zero).
            cost = width * T::Real::HALF * (values[0] + values[1]);
        } else {
            let coefficients = weights(&geometry.nodes[..q], false)?;
            cost = width
                * coefficients
                    .iter()
                    .zip(values)
                    .take(q)
                    .fold(T::Real::ZERO, |sum, (&w, v)| sum + w * v);
            if estimate {
                let higher =
                    Polynomial::new(&geometry.nodes[..=q], |j| values[j])?.integral()? * width;
                qerror = Some(
                    plan.options
                        .quadrature_error
                        .normalized_norm(&[(higher - cost).abs()], &[cost.abs()])?,
                );
            }
        }
    }

    Ok((cost, qerror))
}
