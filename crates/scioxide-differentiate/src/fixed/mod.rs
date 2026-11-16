//! Fixed-step finite differences: the method value, its workspace and kernels.
//!
//! The default steps are listed in the crate docs. Weights use the displacements
//! after rounding. Each operation validates its call inputs once before these
//! kernels; the generated points and callback values are checked as they are
//! produced.

use crate::options::DifferenceShape;
use crate::samples::HessianSamples;
use crate::stencil::{
    MAX_STENCIL_POINTS, PositiveMagnitude, Stencil, axis_stencil_with_nominal, bound_for,
    directional_bounds, directional_stencil_for_points, generated_directional_point,
    mixed_stencil_derivative, nominal_step, normalized_direction, stencil_derivative,
    step_size_limit_error, update_minimum, vacant_stencil,
};
use crate::validation::{
    differentiation_arithmetic_failure, validate_direction, validate_generated_point,
    validate_output, validate_problem, validate_vector_base,
};
use crate::workspace;
use crate::{
    DerivativeRequest, DerivativeSource, DifferenceScheme, DifferenceStep, DifferentiationReport,
    DifferentiationResult, SampleFunction, ZeroDirectionPolicy,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled, validate_then_commit};
use scioxide_core::{
    Bounds, Capacity, DifferentiationLocation, ExecutionContext, ExtendedInterval,
    InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError, SolveScope, SolveState,
};
// Only fixed-step evaluation treats unavailable geometry as a configuration
// error. It names `bounds` when the coordinate is bounded and `x` otherwise.
#[inline]
fn fixed_axis_stencil<S: RealScalar>(
    value: S,
    nominal: S,
    formula: DifferenceScheme,
    derivative_order: usize,
    bound: Option<(Option<S>, Option<S>)>,
    variable: usize,
    output: &mut Stencil<S>,
) -> Result<(), SciOxideError> {
    axis_stencil_with_nominal(value, nominal, formula, derivative_order, bound, output)?.ok_or_else(
        || {
            let bounded = bound.is_some_and(|(lower, upper)| lower.is_some() || upper.is_some());
            SciOxideError::invalid_input(
                if bounded {
                    scioxide_core::argument!("bounds")
                } else {
                    scioxide_core::argument!("x")
                },
                InvalidInputReason::UnsupportedConfiguration,
                alloc::format!(
                    "no representable differentiation stencil exists for variable {variable}"
                ),
            )
        },
    )
}
/// Selects what [`FiniteDifference`] does at a coordinate whose lower and
/// upper bounds are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FixedCoordinatePolicy {
    /// Returns `InvalidInput`, because no stencil fits between equal bounds.
    #[default]
    RequireStencil,
    /// Sets the derivative of such coordinates to zero without sampling them.
    /// Applies to `gradient`, `gradient_into` and `derivative_into`; gradients
    /// require bounds. `derivative` accepts it but takes no bounds, so it has
    /// no effect there. Other operations, including second derivatives, reject it.
    FeasibleZero,
}
/// Computes derivatives by fixed-step forward or central differences.
///
/// Each derivative entry comes from one stencil with the step described in
/// [Conventions](crate#conventions); no error estimate is returned. The
/// callback values must be real. The method value holds settings only: every
/// operation validates them, with the shape taken from its arguments.
///
/// | Operation | Evaluations, forward | Evaluations, central |
/// |---|---|---|
/// | [`derivative`](Self::derivative), [`gradient`](Self::gradient), [`jacobian`](Self::jacobian) | `n + 1` | `2n + 1` |
/// | [`second_derivative`](Self::second_derivative), [`hessian`](Self::hessian) | `1 + 2n + n(n − 1)/2` | `1 + 2n²` |
/// | [`jvp`](Self::jvp) | `2` | `3` |
///
/// The Hessian counts assume interior stencils. A one-sided central diagonal
/// uses three perturbed nodes instead of two. With bounds, a central Hessian
/// costs at most `1 + 3n + 4n(n − 1)` evaluations before sample reuse: up to
/// three diagonal nodes per coordinate and eight non-base nodes per pair. Bit-identical Hessian sample
/// points are evaluated once. A known `f(x)` passed as `base` saves one call.
///
/// With the default steps and a smooth function of moderate size, a central
/// first derivative has an error of order `ε^(2/3)` (`3.7·10⁻¹¹` for `f64`,
/// `2.4·10⁻⁵` for `f32`) and a central second derivative of order `√ε`
/// (`1.5·10⁻⁸`, `3.5·10⁻⁴`). Forward first derivatives have an error of order
/// `√ε` and forward second derivatives of order `ε^(1/3)`.
///
/// The defaults are central differences, the default step, sequential
/// execution, unbounded budgets, no workspace limit,
/// [`FixedCoordinatePolicy::RequireStencil`] and
/// [`ZeroDirectionPolicy::MathematicalZero`].
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::{DifferenceScheme, FiniteDifference};
///
/// // f(x) = x₀² + x₁, so ∇f(1, 2) = (2, 1). The known value f(1, 2) = 3 saves a call.
/// let method = FiniteDifference::default().with_scheme(DifferenceScheme::Forward);
/// let f = |x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0] + x[1];
///     Ok(())
/// };
/// let result = method.gradient(f, &[1.0, 2.0], None, Some(3.0))?;
/// assert_eq!(result.report().evaluations(), 2);
/// let g = result.into_values()?;
/// assert!((g[0] - 2.0).abs() < 1e-7 && (g[1] - 1.0).abs() < 1e-7);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
#[derive(Debug, Clone)]
pub struct FiniteDifference<S: RealScalar> {
    pub(super) scheme: DifferenceScheme,
    pub(super) step: DifferenceStep<S>,
    pub(super) execution: scioxide_core::ExecutionPolicy,
    pub(super) max_workspace_bytes: usize,
    pub(super) budgets: scioxide_core::SolveBudgets,
    pub(super) fixed_coordinates: FixedCoordinatePolicy,
    pub(super) zero_direction: ZeroDirectionPolicy,
}
/// One operation of a [`FiniteDifference`] with its shape, borrowed for a call.
#[derive(Debug)]
struct FixedPlan<'a, S: RealScalar> {
    kind: DifferenceShape,
    inputs: usize,
    outputs: usize,
    len: usize,
    options: &'a FiniteDifference<S>,
}
/// Workspace size of a [`FiniteDifference`] operation, returned by
/// [`FiniteDifferenceWorkspace::workspace_requirements`].
#[derive(Debug, Clone)]
pub struct FiniteDifferenceRequirements<S: RealScalar> {
    kind: DifferenceShape,
    inputs: usize,
    outputs: usize,
    lengths: [usize; 5],
    stencils: [usize; 3],
    reduction: usize,
    axis_nodes: usize,
    bytes: usize,
    limit: usize,
    scalar: core::marker::PhantomData<S>,
}
impl<S: RealScalar> FiniteDifferenceRequirements<S> {
    /// Returns the heap size in bytes of a new workspace for the request.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
    fn new(
        kind: DifferenceShape,
        inputs: usize,
        outputs: usize,
        limit: usize,
    ) -> Result<Self, SciOxideError> {
        let len = crate::validation::shape(kind, inputs, outputs)?;
        let jvp = matches!(kind, DifferenceShape::Jvp(_));
        let vector = matches!(kind, DifferenceShape::Jacobian) || jvp;
        let hessian = matches!(kind, DifferenceShape::Hessian | DifferenceShape::Second);
        let point = if jvp {
            checked_mul(3, inputs, scioxide_core::quantity!("workspace_bytes"))?
        } else {
            inputs
        };
        let lengths = [
            point,
            if vector { outputs } else { 0 },
            if vector { outputs } else { 0 },
            if vector {
                checked_mul(
                    MAX_STENCIL_POINTS,
                    outputs,
                    scioxide_core::quantity!("workspace_bytes"),
                )?
            } else {
                0
            },
            len,
        ];
        let stencils = [
            if hessian || jvp { 0 } else { inputs },
            if hessian { inputs } else { 0 },
            if hessian { inputs } else { 0 },
        ];
        let mut r = Self {
            kind,
            inputs,
            outputs,
            lengths,
            stencils,
            reduction: usize::from(len != 0 && (!jvp || inputs != 0)),
            axis_nodes: if hessian { 2 * MAX_STENCIL_POINTS } else { 0 },
            bytes: 0,
            limit,
            scalar: core::marker::PhantomData,
        };
        r.bytes = FixedBuffers::<S>::default().bytes(Some(&r))?;
        workspace::memory_limit(r.bytes, r.limit)?;
        Ok(r)
    }
}
/// Scratch storage for the operations of [`FiniteDifference`].
///
/// Create it with [`workspace`](Self::workspace), or with `Default` followed by
/// [`try_reserve`](Self::try_reserve). One workspace serves every operation
/// whose requirements it meets. Evaluation does not grow it.
#[derive(Debug, Clone)]
pub struct FiniteDifferenceWorkspace<S: RealScalar> {
    pub(super) data: FixedBuffers<S>,
    state: SolveState,
    required: Option<FiniteDifferenceRequirements<S>>,
}
impl<S: RealScalar> Default for FiniteDifferenceWorkspace<S> {
    fn default() -> Self {
        Self {
            data: FixedBuffers::default(),
            state: SolveState::new(),
            required: None,
        }
    }
}
#[derive(Debug, Clone)]
pub(super) struct FixedBuffers<S: RealScalar> {
    pub(super) input_dimension: usize,
    pub(super) output_dimension: usize,
    pub(super) point: Vec<S>,
    pub(super) base_value: Vec<S>,
    pub(super) value: Vec<S>,
    pub(super) stencil_values: Vec<S>,
    pub(super) derivative: Vec<S>,
    pub(super) stencils: Vec<Stencil<S>>,
    pub(super) first_stencils: Vec<Stencil<S>>,
    pub(super) second_stencils: Vec<Stencil<S>>,
    pub(super) samples: HessianSamples<S>,
    pub(super) reduction: Vec<scioxide_core::RangeAccumulator<S>>,
}
impl<S: RealScalar> Default for FixedBuffers<S> {
    fn default() -> Self {
        Self {
            input_dimension: 0,
            output_dimension: 0,
            point: Vec::new(),
            base_value: Vec::new(),
            value: Vec::new(),
            stencil_values: Vec::new(),
            derivative: Vec::new(),
            stencils: Vec::new(),
            first_stencils: Vec::new(),
            second_stencils: Vec::new(),
            samples: HessianSamples::default(),
            reduction: Vec::new(),
        }
    }
}
impl<S: RealScalar> FixedBuffers<S> {
    fn values(&self) -> [&Vec<S>; 5] {
        [
            &self.point,
            &self.base_value,
            &self.value,
            &self.stencil_values,
            &self.derivative,
        ]
    }
    fn values_mut(&mut self) -> [&mut Vec<S>; 5] {
        [
            &mut self.point,
            &mut self.base_value,
            &mut self.value,
            &mut self.stencil_values,
            &mut self.derivative,
        ]
    }
    fn geometries(&self) -> [&Vec<Stencil<S>>; 3] {
        [&self.stencils, &self.first_stencils, &self.second_stencils]
    }
    fn geometries_mut(&mut self) -> [&mut Vec<Stencil<S>>; 3] {
        [
            &mut self.stencils,
            &mut self.first_stencils,
            &mut self.second_stencils,
        ]
    }
    fn bytes(
        &self,
        required: Option<&FiniteDifferenceRequirements<S>>,
    ) -> Result<usize, SciOxideError> {
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<S>(
                    self.values()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (v.capacity(), required.map_or(0, |r| r.lengths[i]))),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<Stencil<S>>(
                    self.geometries()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (v.capacity(), required.map_or(0, |r| r.stencils[i]))),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<scioxide_core::RangeAccumulator<S>>(
                    [(
                        self.reduction.capacity(),
                        required.map_or(0, |r| r.reduction),
                    )],
                    workspace::BYTES,
                )?,
                match required {
                    Some(r) => self.samples.projected_bytes(r.inputs, r.axis_nodes, 0),
                    None => self.samples.retained_bytes(),
                }?,
            ],
            workspace::BYTES,
        )
    }
    fn configure(&mut self, r: &FiniteDifferenceRequirements<S>) -> Result<(), SciOxideError> {
        for (v, n) in self.values_mut().into_iter().zip(r.lengths) {
            v.resize(n, S::ZERO);
        }
        for (v, n) in self.geometries_mut().into_iter().zip(r.stencils) {
            v.resize(n, vacant_stencil());
        }
        self.reduction
            .resize(r.reduction, scioxide_core::RangeAccumulator::new());
        self.samples.configure(r.inputs, r.axis_nodes, 0)?;
        self.input_dimension = r.inputs;
        self.output_dimension = r.outputs;
        Ok(())
    }
}
impl<S: RealScalar> FiniteDifferenceWorkspace<S> {
    /// Returns the workspace size of `request`, without allocating.
    ///
    /// The size depends on the shape and on the method's
    /// [`max_workspace_bytes`](FiniteDifference::with_max_workspace_bytes)
    /// only.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements(
        request: DerivativeRequest<'_, FiniteDifference<S>>,
    ) -> Result<FiniteDifferenceRequirements<S>, SciOxideError> {
        let (method, shape) = (request.method(), request.shape());
        FiniteDifferenceRequirements::new(
            shape.difference(method.zero_direction),
            shape.inputs(),
            shape.outputs(),
            method.max_workspace_bytes,
        )
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace(
        request: DerivativeRequest<'_, FiniteDifference<S>>,
    ) -> Result<Self, SciOxideError> {
        let mut w = Self::default();
        w.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(w)
    }
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.bytes(None)
    }

    /// Returns the number of inputs of the last request reserved or evaluated,
    /// or zero for a new workspace.
    pub const fn input_dimension(&self) -> usize {
        self.data.input_dimension
    }
    /// Returns the number of callback values of the last request reserved or
    /// evaluated.
    pub const fn output_dimension(&self) -> usize {
        self.data.output_dimension
    }

    /// Reports whether the workspace is large enough for `r`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the method's
    ///   `max_workspace_bytes`.
    pub fn check(&self, r: &FiniteDifferenceRequirements<S>) -> Result<Capacity, SciOxideError> {
        let bytes = self.data.bytes(None)?;
        workspace::check(bytes, r.limit, || {
            Ok(self
                .data
                .values()
                .into_iter()
                .zip(r.lengths)
                .all(|(v, n)| v.capacity() >= n)
                && self
                    .data
                    .geometries()
                    .into_iter()
                    .zip(r.stencils)
                    .all(|(v, n)| v.capacity() >= n)
                && self.data.reduction.capacity() >= r.reduction
                && self.data.samples.has_capacity(r.inputs, r.axis_nodes, 0))
        })
    }

    /// Grows the workspace to meet `r`.
    ///
    /// The byte limit is checked before any buffer grows. If an allocation
    /// fails, buffers grown before it are kept; call [`check`](Self::check)
    /// again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(
        &mut self,
        r: &FiniteDifferenceRequirements<S>,
    ) -> Result<(), SciOxideError> {
        self.required = None;
        let bytes = self.data.bytes(Some(r))?;
        workspace::reserve(
            &mut self.data,
            bytes,
            r.limit,
            |data| {
                for (v, n) in data.values_mut().into_iter().zip(r.lengths) {
                    workspace::reserve_buffer(
                        v,
                        n,
                        S::ZERO,
                        scioxide_core::quantity!("fixed_workspace"),
                    )?;
                }
                for (v, n) in data.geometries_mut().into_iter().zip(r.stencils) {
                    workspace::reserve_buffer(
                        v,
                        n,
                        vacant_stencil(),
                        scioxide_core::quantity!("fixed_workspace"),
                    )?;
                }
                workspace::reserve_buffer(
                    &mut data.reduction,
                    r.reduction,
                    scioxide_core::RangeAccumulator::new(),
                    scioxide_core::quantity!("fixed_workspace"),
                )?;
                data.samples.reserve(r.inputs, r.axis_nodes, 0)?;
                data.configure(r)?;
                Ok(())
            },
            |data| data.bytes(None),
        )?;
        self.required = Some(r.clone());
        Ok(())
    }

    fn prepare(&mut self, plan: &FixedPlan<'_, S>) -> Result<(), SciOxideError> {
        // Requirements depend only on immutable shape and memory limits, never
        // on the point or bounds. Still charge actual retained capacity below.
        if let Some(r) = &self.required
            && r.kind == plan.kind
            && r.inputs == plan.inputs
            && r.outputs == plan.outputs
            && r.limit == plan.options.max_workspace_bytes
            && self.check(r)? == Capacity::Sufficient
        {
            return Ok(());
        }
        let r = plan.requirements()?;
        workspace::prepare(self.check(&r)?, r.bytes, || self.data.bytes(None))?;
        self.data.configure(&r)?;
        self.required = Some(r);
        Ok(())
    }
}

/// The inputs of one call: the point, bounds, known base values and, for a
/// Jacobian-vector product, the direction (empty otherwise).
#[derive(Debug, Clone, Copy)]
struct FixedProblem<'a, S: RealScalar> {
    x: &'a [S],
    bounds: Option<&'a Bounds<S>>,
    base: Option<&'a [S]>,
    direction: &'a [S],
}
impl<'a, S: RealScalar> FixedPlan<'a, S> {
    fn requirements(&self) -> Result<FiniteDifferenceRequirements<S>, SciOxideError> {
        FiniteDifferenceRequirements::new(
            self.kind,
            self.inputs,
            self.outputs,
            self.options.max_workspace_bytes,
        )
    }
    fn validate(&self, p: FixedProblem<'_, S>) -> Result<(), SciOxideError> {
        self.options.step.validate(p.x.len())?;
        validate_problem(p.x, p.bounds, scioxide_core::argument!("x"))?;
        if let Some(base) = p.base {
            validate_vector_base(base, self.outputs, NonFiniteOrigin::Input)?;
        }
        if matches!(self.kind, DifferenceShape::Jvp(_)) {
            validate_direction(p.x, p.direction)?;
        }
        if self.options.fixed_coordinates == FixedCoordinatePolicy::FeasibleZero
            && p.bounds.is_none()
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("bounds"),
                InvalidInputReason::Inconsistent,
                "feasible gradient requires bounds",
            ));
        }
        Ok(())
    }
    fn evaluate<F: SampleFunction<S, S>>(
        &self,
        p: FixedProblem<'_, S>,
        function: F,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.validate(p)?;
        let mut w = FiniteDifferenceWorkspace::default();
        w.try_reserve(&self.requirements()?)?;
        let mut out = try_filled(
            self.len,
            S::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let report = self.evaluate_validated(
            p,
            function,
            &mut out,
            &mut w,
            ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_evaluation(report, out))
    }
    fn evaluate_into<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        p: FixedProblem<'_, S>,
        function: F,
        out: &mut [S],
        w: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate(p)?;
        self.evaluate_validated(p, function, out, w, scope)
    }
    fn evaluate_validated<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        p: FixedProblem<'_, S>,
        mut function: F,
        out: &mut [S],
        w: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        crate::validation::length(self.len, out.len())?;
        w.prepare(self)?;
        let jvp = matches!(self.kind, DifferenceShape::Jvp(_));
        let id = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            if jvp {
                scioxide_core::algorithm_id!("differentiate", "fixed-jvp", 1)
            } else {
                scioxide_core::algorithm_id!("differentiate", "fixed", 1)
            },
            self.len as u128,
            0,
        );
        scope
            .into()
            .run(&mut w.state, id, self.options.budgets, |cx| {
                self.run(p, &mut function, out, &mut w.data, cx)
            })
    }
    fn run<F: SampleFunction<S, S>>(
        &self,
        p: FixedProblem<'_, S>,
        function: &mut F,
        out: &mut [S],
        w: &mut FixedBuffers<S>,
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        let options = self.options;
        match self.kind {
            DifferenceShape::First | DifferenceShape::Gradient => fixed_gradient(
                options,
                |variable| bound_for(p.bounds, variable),
                p.x,
                p.base.map(|b| b[0]),
                function,
                out,
                GradientStorage {
                    point: &mut w.point,
                    stencils: &mut w.stencils,
                    derivative: &mut w.derivative,
                    reduction: &mut w.reduction,
                },
                options.fixed_coordinates == FixedCoordinatePolicy::FeasibleZero,
                cx,
            ),
            DifferenceShape::Second | DifferenceShape::Hessian => fixed_hessian(
                options,
                p.bounds,
                p.x,
                p.base.map(|b| b[0]),
                function,
                out,
                w,
                cx,
            ),
            DifferenceShape::Jacobian => {
                fixed_jacobian(options, p.bounds, p.x, p.base, function, out, w, cx)
            }
            DifferenceShape::Jvp(policy) => fixed_jvp(
                options,
                policy,
                p.bounds,
                p.x,
                p.direction,
                p.base,
                function,
                out,
                w,
                cx,
            ),
        }
    }
}
impl<S: RealScalar> Default for FiniteDifference<S> {
    fn default() -> Self {
        Self {
            scheme: DifferenceScheme::Central,
            step: DifferenceStep::default(),
            execution: scioxide_core::ExecutionPolicy::Sequential,
            max_workspace_bytes: usize::MAX,
            budgets: scioxide_core::SolveBudgets::unbounded(),
            fixed_coordinates: FixedCoordinatePolicy::default(),
            zero_direction: ZeroDirectionPolicy::MathematicalZero,
        }
    }
}
impl<S: RealScalar> FiniteDifference<S> {
    /// Sets the difference scheme.
    pub fn with_scheme(mut self, scheme: DifferenceScheme) -> Self {
        self.scheme = scheme;
        self
    }
    /// Sets the step, replacing any earlier step selection.
    pub fn with_step(mut self, step: DifferenceStep<S>) -> Self {
        self.step = step;
        self
    }
    /// Returns the difference scheme.
    pub const fn scheme(&self) -> DifferenceScheme {
        self.scheme
    }
    /// Returns the step selection.
    pub const fn step(&self) -> &DifferenceStep<S> {
        &self.step
    }
    /// Sets the execution policy for [`Parallel`](crate::Parallel) callbacks.
    pub const fn with_execution(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Limits the heap size of the workspace in bytes, including capacity left
    /// from earlier operations.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Sets the evaluation and iteration budgets of each evaluation. Limits of
    /// an enclosing solve still apply.
    pub const fn with_budgets(mut self, budgets: scioxide_core::SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Sets the behavior at coordinates with equal lower and upper bounds.
    pub const fn with_fixed_coordinates(mut self, policy: FixedCoordinatePolicy) -> Self {
        self.fixed_coordinates = policy;
        self
    }
    /// Sets what [`jvp`](Self::jvp) does when every component of the direction
    /// is zero. Other operations require the default.
    pub const fn with_zero_direction(mut self, policy: ZeroDirectionPolicy) -> Self {
        self.zero_direction = policy;
        self
    }
    pub(super) fn nominal(&self, x: S, index: usize, order: usize) -> S {
        self.step
            .magnitude(x, index, nominal_step(x, self.scheme, order))
    }
    // Validates the settings for one operation; every call runs it.
    fn plan(
        &self,
        kind: DifferenceShape,
        inputs: usize,
        outputs: usize,
    ) -> Result<FixedPlan<'_, S>, SciOxideError> {
        let len = crate::validation::shape(kind, inputs, outputs)?;
        if self.fixed_coordinates == FixedCoordinatePolicy::FeasibleZero
            && kind != DifferenceShape::Gradient
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "feasible-zero coordinates require a gradient",
            ));
        }
        if !matches!(kind, DifferenceShape::Jvp(_)) {
            self.require_mathematical_zero()?;
        }
        Ok(FixedPlan {
            kind,
            inputs,
            outputs,
            len,
            options: self,
        })
    }
    fn require_mathematical_zero(&self) -> Result<(), SciOxideError> {
        if self.zero_direction == ZeroDirectionPolicy::MathematicalZero {
            Ok(())
        } else {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "a zero-direction policy applies only to Jacobian-vector products",
            ))
        }
    }

    /// Computes `f′(x)` of `f: ℝ → ℝ` and returns it with the report.
    ///
    /// The derivative uses the stencil of [`derivative_into`](Self::derivative_into)
    /// without bounds and allocates the one-element result. The evaluation
    /// runs with [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`derivative_into`](Self::derivative_into).
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = sin x, so f′(0) = 1.
    /// let result = FiniteDifference::default().derivative(|x: f64| Ok(x.sin()), 0.0)?;
    /// assert_eq!(result.report().evaluations(), 3);
    /// assert!((result.into_values()?[0] - 1.0).abs() < 1e-10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative<F: FnMut(S) -> Result<S, SciOxideError>>(
        &self,
        function: F,
        x: S,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        let mut values = try_filled(1, S::ZERO, scioxide_core::quantity!("derivative_output"))?;
        let report = self.derivative_into(
            function,
            x,
            ExtendedInterval::entire(),
            None,
            &mut values[0],
            ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_evaluation(report, values))
    }
    /// Computes `f″(x)` of `f: ℝ → ℝ` and returns it with the report.
    ///
    /// It is the [`hessian`](Self::hessian) of the one-coordinate function,
    /// without bounds. A central second difference uses the nodes
    /// `x − h, x, x + h` and a forward one `x, x + h, x + 2h`, so both cost
    /// three evaluations.
    ///
    /// # Errors
    ///
    /// - The errors of [`hessian`](Self::hessian).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::{DifferenceScheme, FiniteDifference};
    ///
    /// // f(x) = x³, so f″(1) = 6.
    /// let method = FiniteDifference::default().with_scheme(DifferenceScheme::Forward);
    /// let result = method.second_derivative(|x: f64| Ok(x * x * x), 1.0)?;
    /// assert_eq!(result.report().evaluations(), 3);
    /// assert!((result.into_values()?[0] - 6.0).abs() < 1e-3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn second_derivative<F: FnMut(S) -> Result<S, SciOxideError>>(
        &self,
        mut function: F,
        x: S,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.hessian(
            |p: &[S], y: &mut [S]| {
                y[0] = function(p[0])?;
                Ok(())
            },
            &[x],
            None,
            None,
        )
    }
    /// Computes `f′(x)` of `f: ℝ → ℝ` into `out` without a workspace.
    ///
    /// Every sample lies in `bounds`, and a known `base = f(x)` is not
    /// evaluated again. The stencil, step, budgets and fixed-coordinate policy
    /// are those of [`gradient_into`](Self::gradient_into) with one coordinate.
    /// The evaluation uses only stack storage, so it suits repeated calls, for
    /// example inside a one-dimensional solver. `scope` is an
    /// [`ExecutionContext`] or a child [`SolveScope`]. `out` is written only if
    /// the report has output available.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`bounds`, `Empty`) if `bounds` is empty.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `InvalidInput` (`bounds` or `x`, `UnsupportedConfiguration`) if no
    ///   stencil with the step fits at `x`.
    /// - `InvalidInput` (`self`, `Inconsistent`) if the zero-direction policy is
    ///   not the default.
    /// - `NonFinite` (origin `Input`) if `x` or `base` is NaN or infinite.
    /// - `DimensionMismatch` if the step has per-coordinate values for more or
    ///   fewer than one coordinate.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value
    ///   of `function`, or `NumericalInstability` for a sample point or stencil
    ///   sum outside the finite range; it records the sample location and the
    ///   evaluation count.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, ExtendedInterval};
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = x² on [0, 1]: at the upper bound x = 1 the stencil is one-sided; f′(1) = 2.
    /// let bounds = ExtendedInterval::new(Some(0.0), Some(1.0))?;
    /// let (mut slope, context) = (0.0, ExecutionContext::sequential());
    /// let method = FiniteDifference::default();
    /// let f = |x: f64| Ok(x * x);
    /// let report = method.derivative_into(f, 1.0, bounds, None, &mut slope, context)?;
    /// assert!(report.output_available() && (slope - 2.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative_into<'scope, 'capability, F: FnMut(S) -> Result<S, SciOxideError>>(
        &self,
        mut function: F,
        x: S,
        bounds: ExtendedInterval<S>,
        base: Option<S>,
        out: &mut S,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.require_mathematical_zero()?;
        let bound = bounds.bounds().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("bounds"),
                InvalidInputReason::Empty,
                "bounds must contain the differentiation point",
            )
        })?;
        self.step.validate(1)?;
        validate_problem(&[x], None, scioxide_core::argument!("x"))?;
        if !bounds.contains(x) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                InvalidInputReason::OutOfRange,
                "input coordinates must lie within their bounds",
            ));
        }
        if let Some(base) = base {
            validate_vector_base(&[base], 1, NonFiniteOrigin::Input)?;
        }
        let bound = (!bounds.is_entire()).then_some(bound);
        let mut point = [x];
        let mut stencils = [vacant_stencil()];
        let mut derivative = [S::ZERO];
        let mut reduction = [scioxide_core::RangeAccumulator::new()];
        let mut sample = |p: &[S], y: &mut [S]| -> Result<(), SciOxideError> {
            y[0] = function(p[0])?;
            Ok(())
        };
        let id = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("differentiate", "fixed", 1),
            1,
            0,
        );
        scope
            .into()
            .run(&mut SolveState::new(), id, self.budgets, |cx| {
                fixed_gradient(
                    self,
                    |_| bound,
                    &[x],
                    base,
                    &mut sample,
                    core::slice::from_mut(out),
                    GradientStorage {
                        point: &mut point,
                        stencils: &mut stencils,
                        derivative: &mut derivative,
                        reduction: &mut reduction,
                    },
                    self.fixed_coordinates == FixedCoordinatePolicy::FeasibleZero,
                    cx,
                )
            })
    }

    /// Computes the gradient `∇f(x)` of `f: ℝⁿ → ℝ` and returns it with the
    /// report.
    ///
    /// `bounds` restrict every sample point; near a bound a stencil becomes
    /// one-sided (see [Conventions](crate#conventions)). `base` is the known
    /// value `f(x)`, which is then not evaluated again. The evaluation runs with
    /// [`ExecutionContext::sequential_native`] and allocates a workspace and the
    /// result.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or the result cannot be
    ///   allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = x₀ x₁, so ∇f(2, 3) = (3, 2).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[1];
    ///     Ok(())
    /// };
    /// let g = FiniteDifference::default().gradient(f, &[2.0, 3.0], None, None)?.into_values()?;
    /// assert!((g[0] - 3.0).abs() < 1e-9 && (g[1] - 2.0).abs() < 1e-9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn gradient<F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<S>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x.len(), 1)?.evaluate(
            problem(x, bounds, base.as_ref().map(core::slice::from_ref)),
            function,
        )
    }
    /// Computes the gradient `∇f(x)` of `f: ℝⁿ → ℝ` into `out`, using
    /// `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`] or a child [`SolveScope`];
    /// [`Parallel`](crate::Parallel) callbacks run on its executor. With a
    /// sufficient workspace the evaluation does not allocate. `out` is written
    /// only if the report has output available.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds`, `out` or per-coordinate steps do not
    ///   have `n` entries, or `workspace` is too small.
    /// - `NonFinite` (origin `Input`) if `x` or `base` contains NaN or infinity.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `InvalidInput` (`bounds` or `x`, `UnsupportedConfiguration`) if no
    ///   stencil fits at a coordinate; `bounds` when the coordinate is bounded.
    /// - `InvalidInput` (`bounds`, `Inconsistent`) if
    ///   [`FixedCoordinatePolicy::FeasibleZero`] is selected without bounds.
    /// - `InvalidInput` (`self`, `Inconsistent`) if the zero-direction policy is
    ///   not the default.
    /// - `DifferentiationFailure` as in [`derivative_into`](Self::derivative_into).
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller output and borrowed resources remain explicit"
    )]
    pub fn gradient_into<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<S>,
        out: &mut [S],
        workspace: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x.len(), 1)?
            .evaluate_into(
                problem(x, bounds, base.as_ref().map(core::slice::from_ref)),
                function,
                out,
                workspace,
                scope,
            )
    }
    /// Computes the Hessian of `f: ℝⁿ → ℝ`, row-major `n × n`, and returns it
    /// with the report.
    ///
    /// Diagonal entries use a second-difference stencil and mixed entries the
    /// tensor product of two first-difference stencils with the
    /// second-derivative step; only the upper triangle is evaluated, so the
    /// result is exactly symmetric. Arguments are as in
    /// [`gradient`](Self::gradient).
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient), except that
    ///   [`FixedCoordinatePolicy::FeasibleZero`] is rejected with
    ///   `InvalidInput` (`self`, `Inconsistent`).
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = x₀² x₁ has the Hessian [[2x₁, 2x₀], [2x₀, 0]] = [[4, 2], [2, 0]] at (1, 2).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[0] * x[1];
    ///     Ok(())
    /// };
    /// let h = FiniteDifference::default().hessian(f, &[1.0, 2.0], None, None)?.into_values()?;
    /// for (value, expected) in h.iter().zip([4.0, 2.0, 2.0, 0.0]) {
    ///     assert!((value - expected).abs() < 1e-6);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn hessian<F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<S>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Hessian, x.len(), 1)?.evaluate(
            problem(x, bounds, base.as_ref().map(core::slice::from_ref)),
            function,
        )
    }
    /// Computes the Hessian of `f: ℝⁿ → ℝ` into `out`, using `workspace` for
    /// scratch.
    ///
    /// Arguments are as in [`gradient_into`](Self::gradient_into); `out` has
    /// `n²` entries.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into), except that
    ///   [`FixedCoordinatePolicy::FeasibleZero`] is rejected with
    ///   `InvalidInput` (`self`, `Inconsistent`).
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller output and borrowed resources remain explicit"
    )]
    pub fn hessian_into<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<S>,
        out: &mut [S],
        workspace: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Hessian, x.len(), 1)?
            .evaluate_into(
                problem(x, bounds, base.as_ref().map(core::slice::from_ref)),
                function,
                out,
                workspace,
                scope,
            )
    }
    /// Computes the Jacobian of `f: ℝⁿ → ℝᵐ` with `m = outputs`, row-major
    /// `m × n`, and returns it with the report.
    ///
    /// Entry `i·n + j` is `∂fᵢ/∂xⱼ`. `base` is the known value `f(x)` with `m`
    /// entries. Other arguments are as in [`gradient`](Self::gradient).
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient), except that
    ///   [`FixedCoordinatePolicy::FeasibleZero`] is rejected with
    ///   `InvalidInput` (`self`, `Inconsistent`).
    /// - `DimensionMismatch` if `base` does not have `m` entries.
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = (x₀ x₁, x₀ + x₁) has the Jacobian [[x₁, x₀], [1, 1]] = [[2, 1], [1, 1]].
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[1];
    ///     y[1] = x[0] + x[1];
    ///     Ok(())
    /// };
    /// let j = FiniteDifference::default().jacobian(f, &[1.0, 2.0], 2, None, None)?.into_values()?;
    /// for (value, expected) in j.iter().zip([2.0, 1.0, 1.0, 1.0]) {
    ///     assert!((value - expected).abs() < 1e-9);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jacobian<F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
        base: Option<&[S]>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x.len(), outputs)?
            .evaluate(problem(x, bounds, base), function)
    }
    /// Computes the Jacobian of `f: ℝⁿ → ℝᵐ` with `m = outputs` into `out`,
    /// using `workspace` for scratch.
    ///
    /// Arguments are as in [`gradient_into`](Self::gradient_into); `out` has
    /// `m·n` entries and `base` has `m`.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into), except that
    ///   [`FixedCoordinatePolicy::FeasibleZero`] is rejected with
    ///   `InvalidInput` (`self`, `Inconsistent`).
    /// - `DimensionMismatch` if `base` does not have `m` entries.
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller output and borrowed resources remain explicit"
    )]
    pub fn jacobian_into<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
        base: Option<&[S]>,
        out: &mut [S],
        workspace: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x.len(), outputs)?
            .evaluate_into(problem(x, bounds, base), function, out, workspace, scope)
    }
    /// Computes the Jacobian-vector product `J(x) v` of `f: ℝⁿ → ℝᵐ` with
    /// `v = direction` and `m = outputs`, and returns it with the report.
    ///
    /// One stencil along `v` replaces the `n` stencils of a full Jacobian. The
    /// step along `v` is `α = c / maxᵢ(|vᵢ| / max(1, |xᵢ|))`, with the factor
    /// `c` of the default step, so that no coordinate moves farther than its
    /// own step. A zero direction follows
    /// [`with_zero_direction`](Self::with_zero_direction). Other arguments are
    /// as in [`jacobian`](Self::jacobian).
    ///
    /// # Errors
    ///
    /// - The errors of [`jacobian`](Self::jacobian), except the zero-direction
    ///   rule.
    /// - `DimensionMismatch` if `direction` does not have `n` entries.
    /// - `NonFinite` (origin `Input`) if `direction` contains NaN or infinity.
    /// - `DifferentiationFailure` whose `source()` is `ConvergenceFailure`
    ///   (`StepSizeLimit`) if no step along `direction` gives distinct finite
    ///   points inside the bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = (x₀ + x₁, 2x₀ − x₁) is linear, so J v = (1, 2) for v = (1, 0).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] + x[1];
    ///     y[1] = 2.0 * x[0] - x[1];
    ///     Ok(())
    /// };
    /// // f(1, 2) = (3, 0) is known, so the central difference makes two calls.
    /// let method = FiniteDifference::default();
    /// let result = method.jvp(f, &[1.0, 2.0], &[1.0, 0.0], 2, None, Some(&[3.0, 0.0]))?;
    /// assert_eq!(result.report().evaluations(), 2);
    /// let jv = result.into_values()?;
    /// assert!((jv[0] - 1.0).abs() < 1e-8 && (jv[1] - 2.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jvp<F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
        base: Option<&[S]>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Jvp(self.zero_direction), x.len(), outputs)?
            .evaluate(
                FixedProblem {
                    direction,
                    ..problem(x, bounds, base)
                },
                function,
            )
    }
    /// Computes the Jacobian-vector product `J(x) v` with `v = direction` into
    /// `out`, using `workspace` for scratch.
    ///
    /// The number of outputs `m` is `out.len()`. Other arguments are as in
    /// [`jacobian_into`](Self::jacobian_into).
    ///
    /// # Errors
    ///
    /// - The errors of [`jvp`](Self::jvp), except allocation.
    /// - `DimensionMismatch` if `workspace` is too small.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller output and borrowed resources remain explicit"
    )]
    pub fn jvp_into<'scope, 'capability, F: SampleFunction<S, S>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<&[S]>,
        out: &mut [S],
        workspace: &mut FiniteDifferenceWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(
            DifferenceShape::Jvp(self.zero_direction),
            x.len(),
            out.len(),
        )?
        .evaluate_into(
            FixedProblem {
                direction,
                ..problem(x, bounds, base)
            },
            function,
            out,
            workspace,
            scope,
        )
    }
}
fn problem<'a, S: RealScalar>(
    x: &'a [S],
    bounds: Option<&'a Bounds<S>>,
    base: Option<&'a [S]>,
) -> FixedProblem<'a, S> {
    FixedProblem {
        x,
        bounds,
        base,
        direction: &[],
    }
}
struct FixedEvaluation<'a, 'b, 'c> {
    context: &'a mut scioxide_core::SolveContext<'b, 'c>,
    policy: scioxide_core::ExecutionPolicy,
    report: DifferentiationReport,
}
macro_rules! admitted {
    ($value:expr, $work:expr) => {
        match $value? {
            Some(value) => value,
            None => return Ok($work.report),
        }
    };
}
fn evaluate_scalar<S: RealScalar, F>(
    function: &mut F,
    point: &[S],
    location: DifferentiationLocation,
    work: &mut FixedEvaluation<'_, '_, '_>,
) -> Result<Option<S>, SciOxideError>
where
    F: SampleFunction<S, S>,
{
    validate_fixed_point(point, location, work.report.evaluations())?;
    let mut value = [S::ZERO];
    let complete = work.report.invoke(
        point,
        &mut value,
        function,
        location,
        work.context,
        work.policy,
    )?;
    Ok(complete.then_some(value[0]))
}
fn evaluate_vector<S: RealScalar, F>(
    function: &mut F,
    point: &[S],
    output: &mut [S],
    location: DifferentiationLocation,
    work: &mut FixedEvaluation<'_, '_, '_>,
) -> Result<Option<()>, SciOxideError>
where
    F: SampleFunction<S, S>,
{
    validate_fixed_point(point, location, work.report.evaluations())?;
    Ok(work
        .report
        .invoke(point, output, function, location, work.context, work.policy)?
        .then_some(()))
}
// Fixed kernels restore an already validated input and perturb only the axes
// named by the sample location. Directional points can change every coordinate.
#[inline]
fn validate_fixed_point<S: RealScalar>(
    point: &[S],
    location: DifferentiationLocation,
    evaluations: usize,
) -> Result<(), SciOxideError> {
    use scioxide_core::DifferentiationSite;
    let finite = match location.site() {
        DifferentiationSite::BasePoint => true,
        DifferentiationSite::Variable(i) => point[i].is_finite(),
        DifferentiationSite::Variables(i, j) => point[i].is_finite() && point[j].is_finite(),
        _ => return validate_generated_point(point, location, evaluations),
    };
    if finite {
        Ok(())
    } else {
        // Keep the existing first-invalid-coordinate diagnostics on failure.
        validate_generated_point(point, location, evaluations)
    }
}
fn prepare_scalar_base<S: RealScalar, F>(
    function: &mut F,
    x: &[S],
    base: Option<S>,
    work: &mut FixedEvaluation<'_, '_, '_>,
) -> Result<Option<S>, SciOxideError>
where
    F: SampleFunction<S, S>,
{
    match base {
        Some(value) => Ok(Some(value)),
        None => evaluate_scalar(function, x, DifferentiationLocation::base_point(), work),
    }
}
fn prepare_vector_base<S: RealScalar, F>(
    function: &mut F,
    x: &[S],
    base: Option<&[S]>,
    base_value: &mut [S],
    work: &mut FixedEvaluation<'_, '_, '_>,
    location: DifferentiationLocation,
) -> Result<Option<()>, SciOxideError>
where
    F: SampleFunction<S, S>,
{
    if let Some(base) = base {
        base_value.copy_from_slice(base);
        Ok(Some(()))
    } else {
        evaluate_vector(function, x, base_value, location, work)
    }
}

fn fixed_coordinate<S: RealScalar>(bound: Option<(Option<S>, Option<S>)>) -> bool {
    matches!(bound, Some((Some(lower), Some(upper))) if lower == upper)
}

/// Borrowed gradient staging: the perturbed point, one stencil and one staged
/// derivative per variable, and the derivative reduction. Workspace and stack
/// evaluations share [`fixed_gradient`] through it.
struct GradientStorage<'a, S: RealScalar> {
    point: &'a mut [S],
    stencils: &'a mut [Stencil<S>],
    derivative: &'a mut [S],
    reduction: &'a mut [scioxide_core::RangeAccumulator<S>],
}

#[expect(
    clippy::too_many_arguments,
    reason = "numerical inputs, caller-owned staging and execution capabilities stay explicit"
)]
#[expect(
    clippy::needless_range_loop,
    reason = "coupled numerical coordinates and mutable scratch require indexed traversal"
)]
fn fixed_gradient<S, F>(
    options: &FiniteDifference<S>,
    bounds: impl Fn(usize) -> Option<(Option<S>, Option<S>)>,
    x: &[S],
    base: Option<S>,
    function: &mut F,
    output: &mut [S],
    workspace: GradientStorage<'_, S>,
    zero_fixed: bool,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<DifferentiationReport, SciOxideError>
where
    S: RealScalar,
    F: SampleFunction<S, S>,
{
    let scheme = options.scheme;
    // Invariant for the loop below: `workspace.point` equals `x` except while
    // one coordinate is perturbed. That coordinate is restored before the
    // evaluation result is inspected, so an error cannot leave it dirty.
    workspace.point.copy_from_slice(x);
    for (variable, &coordinate) in x.iter().enumerate() {
        if zero_fixed && fixed_coordinate(bounds(variable)) {
            continue;
        }
        fixed_axis_stencil(
            coordinate,
            options.nominal(coordinate, variable, 1),
            scheme,
            1,
            bounds(variable),
            variable,
            &mut workspace.stencils[variable],
        )?;
    }
    let mut evaluations = FixedEvaluation {
        context,
        policy: options.execution,
        report: DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(scheme),
            output.len(),
        ),
    };
    if output.is_empty() {
        evaluations.report.poll(evaluations.context)?;
        return Ok(evaluations.report);
    }

    let base = admitted!(
        prepare_scalar_base(function, x, base, &mut evaluations),
        evaluations
    );

    for variable in 0..x.len() {
        if !evaluations.report.iteration(evaluations.context)? {
            return Ok(evaluations.report);
        }
        if zero_fixed && fixed_coordinate(bounds(variable)) {
            workspace.derivative[variable] = S::ZERO;
            evaluations.report.completed_entries += 1;
            continue;
        }
        let stencil = workspace.stencils[variable];
        let mut samples = [S::ZERO; MAX_STENCIL_POINTS];
        for point_index in 0..stencil.len {
            let offset = stencil.offsets[point_index];
            // The zero-offset node of a central or one-sided stencil is f(x), which is
            // already known, so it costs no callback.
            samples[point_index] = if offset == S::ZERO {
                base
            } else {
                workspace.point[variable] = x[variable] + offset;
                let evaluation = evaluate_scalar(
                    function,
                    workspace.point,
                    DifferentiationLocation::variable(variable),
                    &mut evaluations,
                );
                workspace.point[variable] = x[variable];
                admitted!(evaluation, evaluations)
            };
        }
        let derivative = stencil_derivative(
            stencil,
            &samples[..stencil.len],
            base,
            &mut workspace.reduction[0],
        )
        .map_err(|error| {
            differentiation_arithmetic_failure(
                DifferentiationLocation::variable(variable),
                evaluations.report.evaluations(),
                error,
            )
        })?;
        workspace.derivative[variable] = derivative;
        evaluations.report.completed_entries += 1;
    }
    // Results are staged in the workspace and copied only after every entry is
    // known to be finite, so `output` is untouched on any error.
    if !evaluations.report.poll(evaluations.context)? {
        return Ok(evaluations.report);
    }
    validate_then_commit(&workspace.derivative[..x.len()], output, |values| {
        validate_output(values, "gradient output is non-finite")
    })?;
    Ok(evaluations.report)
}

/// Computes a row-major finite-difference Jacobian into `output`.
///
/// The output length must be `output_dimension * x.len()`. The output remains
/// unchanged if any error is returned.
///
/// # Errors
///
/// Returns structured input, workspace, callback, non-finite, or numerical errors.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical kernel keeps problem data and execution explicit"
)]
fn fixed_jacobian<S, F>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    base: Option<&[S]>,
    function: &mut F,
    output: &mut [S],
    workspace: &mut FixedBuffers<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<DifferentiationReport, SciOxideError>
where
    S: RealScalar,
    F: SampleFunction<S, S>,
{
    let scheme = options.scheme;
    let m = workspace.output_dimension;
    let mut evaluations = FixedEvaluation {
        context,
        policy: options.execution,
        report: DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(scheme),
            output.len(),
        ),
    };
    if output.is_empty() {
        evaluations.report.poll(evaluations.context)?;
        return Ok(evaluations.report);
    }

    workspace.point.copy_from_slice(x);
    for (variable, &coordinate) in x.iter().enumerate() {
        fixed_axis_stencil(
            coordinate,
            options.nominal(coordinate, variable, 1),
            scheme,
            1,
            bound_for(bounds, variable),
            variable,
            &mut workspace.stencils[variable],
        )?;
    }

    admitted!(
        prepare_vector_base(
            function,
            x,
            base,
            &mut workspace.base_value,
            &mut evaluations,
            DifferentiationLocation::base_point(),
        ),
        evaluations
    );

    for variable in 0..x.len() {
        if !evaluations.report.iteration(evaluations.context)? {
            return Ok(evaluations.report);
        }
        let stencil = workspace.stencils[variable];
        let first = stencil.first_difference();
        for point_index in 0..stencil.len {
            let offset = stencil.offsets[point_index];
            if offset == S::ZERO {
                if first.is_some() {
                    continue;
                }
                workspace.value.copy_from_slice(&workspace.base_value);
            } else {
                workspace.point[variable] = x[variable] + offset;
                let evaluation = evaluate_vector(
                    function,
                    &workspace.point,
                    &mut workspace.value,
                    DifferentiationLocation::variable(variable),
                    &mut evaluations,
                );
                workspace.point[variable] = x[variable];
                admitted!(evaluation, evaluations);
            }
            // Transpose into component-major rows of stride MAX_STENCIL_POINTS so that
            // each output component's samples form one contiguous stencil slice.
            if first.is_none() {
                for component in 0..m {
                    workspace.stencil_values[component * MAX_STENCIL_POINTS + point_index] =
                        workspace.value[component];
                }
            }
        }
        for component in 0..m {
            let index = component * x.len() + variable;
            let start = component * MAX_STENCIL_POINTS;
            let pair = [workspace.base_value[component], workspace.value[component]];
            let derivative = if let Some(first) = first {
                match first.native(pair[0], pair[1]) {
                    Some(value) => Ok(value),
                    None => {
                        stencil_derivative(stencil, &pair, pair[0], &mut workspace.reduction[0])
                    }
                }
            } else {
                stencil_derivative(
                    stencil,
                    &workspace.stencil_values[start..start + stencil.len],
                    workspace.base_value[component],
                    &mut workspace.reduction[0],
                )
            };
            workspace.derivative[index] = derivative.map_err(|error| {
                differentiation_arithmetic_failure(
                    DifferentiationLocation::variable(variable).with_component(component),
                    evaluations.report.evaluations(),
                    error,
                )
            })?;
            evaluations.report.completed_entries += 1;
        }
    }
    if !evaluations.report.poll(evaluations.context)? {
        return Ok(evaluations.report);
    }
    validate_then_commit(&workspace.derivative[..output.len()], output, |values| {
        validate_output(values, "Jacobian output is non-finite")
    })?;
    Ok(evaluations.report)
}

fn evaluate_hessian_point<S: RealScalar, F>(
    function: &mut F,
    x: &[S],
    workspace: &mut FixedBuffers<S>,
    first: (usize, S),
    second: Option<(usize, S)>,
    base: S,
    evaluations: &mut FixedEvaluation<'_, '_, '_>,
) -> Result<Option<S>, SciOxideError>
where
    F: SampleFunction<S, S>,
{
    if first.1 == S::ZERO && second.is_none_or(|(_, offset)| offset == S::ZERO) {
        return Ok(Some(base));
    }
    // Only the named axes can change. Classify their actual IEEE coordinates
    // before copying the whole point; a hit needs no point construction.
    let first_node = x[first.0] + first.1;
    let cached = match second {
        None => workspace.samples.find_axis(first.0, first_node),
        Some((axis, offset)) => {
            let second_node = x[axis] + offset;
            if crate::samples::same_bits(first_node, x[first.0]) {
                workspace.samples.find_axis(axis, second_node)
            } else if crate::samples::same_bits(second_node, x[axis]) {
                workspace.samples.find_axis(first.0, first_node)
            } else {
                None
            }
        }
    };
    if let Some(value) = cached {
        return Ok(Some(value));
    }
    // Mixed points perturb two coordinates, so the whole point is reset from
    // `x` rather than restoring individual coordinates afterwards.
    workspace.point.copy_from_slice(x);
    workspace.point[first.0] = x[first.0] + first.1;
    let location = if let Some((variable, offset)) = second {
        workspace.point[variable] = x[variable] + offset;
        DifferentiationLocation::variables(first.0, variable)
    } else {
        DifferentiationLocation::variable(first.0)
    };
    // Distinct tensor nodes visit a genuinely mixed point only once. Cache
    // lookup/storage is needed only where the tensor meets a base or axis.
    let retain = second.is_none_or(|(variable, _)| {
        crate::samples::same_bits(workspace.point[first.0], x[first.0])
            || crate::samples::same_bits(workspace.point[variable], x[variable])
    });
    let value = evaluate_scalar(function, &workspace.point, location, evaluations)?;
    if retain && let Some(value) = value {
        workspace.samples.insert_axes(
            x,
            &workspace.point,
            first.0,
            second.map(|(i, _)| i),
            value,
        )?;
    }
    Ok(value)
}

/// Computes a row-major finite-difference Hessian into `output`.
///
/// Within one call, bit-identical points reuse the first checked value; points
/// that differ only in the sign of zero are distinct. A stateful callback
/// therefore runs once at each distinct point, and only actual calls count as
/// evaluations. The storage is reported by
/// [`FiniteDifferenceWorkspace::workspace_requirements`].
///
/// The output remains unchanged if any error is returned.
///
/// # Errors
///
/// Returns structured input, workspace, callback, non-finite, or numerical errors.
#[expect(
    clippy::needless_range_loop,
    reason = "coupled numerical coordinates and mutable scratch require indexed traversal"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical kernel keeps problem data and execution explicit"
)]
fn fixed_hessian<S, F>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    base: Option<S>,
    function: &mut F,
    output: &mut [S],
    workspace: &mut FixedBuffers<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<DifferentiationReport, SciOxideError>
where
    S: RealScalar,
    F: SampleFunction<S, S>,
{
    workspace.samples.clear();
    let scheme = options.scheme;
    for (variable, &coordinate) in x.iter().enumerate() {
        fixed_axis_stencil(
            coordinate,
            options.nominal(coordinate, variable, 2),
            scheme,
            2,
            bound_for(bounds, variable),
            variable,
            &mut workspace.second_stencils[variable],
        )?;
        fixed_axis_stencil(
            coordinate,
            options.nominal(coordinate, variable, 2),
            scheme,
            1,
            bound_for(bounds, variable),
            variable,
            &mut workspace.first_stencils[variable],
        )?;
    }
    let mut evaluations = FixedEvaluation {
        context,
        policy: options.execution,
        report: DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(scheme),
            output.len(),
        ),
    };
    if output.is_empty() {
        evaluations.report.poll(evaluations.context)?;
        return Ok(evaluations.report);
    }

    let base = admitted!(
        prepare_scalar_base(function, x, base, &mut evaluations),
        evaluations
    );

    for variable in 0..x.len() {
        if !evaluations.report.iteration(evaluations.context)? {
            return Ok(evaluations.report);
        }
        let stencil = workspace.second_stencils[variable];
        let mut samples = [S::ZERO; MAX_STENCIL_POINTS];
        for point_index in 0..stencil.len {
            samples[point_index] = admitted!(
                evaluate_hessian_point(
                    function,
                    x,
                    workspace,
                    (variable, stencil.offsets[point_index]),
                    None,
                    base,
                    &mut evaluations,
                ),
                evaluations
            );
        }
        if !evaluations.report.poll(evaluations.context)? {
            return Ok(evaluations.report);
        }
        workspace.derivative[variable * x.len() + variable] = stencil_derivative(
            stencil,
            &samples[..stencil.len],
            base,
            &mut workspace.reduction[0],
        )
        .map_err(|error| {
            differentiation_arithmetic_failure(
                DifferentiationLocation::variables(variable, variable),
                evaluations.report.evaluations(),
                error,
            )
        })?;
        evaluations.report.completed_entries += 1;
    }

    // Off-diagonal entries use the tensor product of two first-derivative
    // stencils. The step is the second-derivative one because dividing by
    // h_first * h_second amplifies rounding error like eps / h^2. Only the upper
    // triangle is evaluated and then mirrored, which halves the callbacks and
    // makes the result exactly symmetric.
    for first in 0..x.len() {
        let first_stencil = workspace.first_stencils[first];
        for second in (first + 1)..x.len() {
            workspace.samples.clear_pair();
            if !evaluations.report.iteration(evaluations.context)? {
                return Ok(evaluations.report);
            }
            let second_stencil = workspace.first_stencils[second];
            let mut samples = [S::ZERO; MAX_STENCIL_POINTS * MAX_STENCIL_POINTS];
            for first_point in 0..first_stencil.len {
                for second_point in 0..second_stencil.len {
                    let first_offset = first_stencil.offsets[first_point];
                    let second_offset = second_stencil.offsets[second_point];
                    let evaluated = if first_offset == S::ZERO && second_offset == S::ZERO {
                        base
                    } else {
                        admitted!(
                            evaluate_hessian_point(
                                function,
                                x,
                                workspace,
                                (first, first_offset),
                                Some((second, second_offset)),
                                base,
                                &mut evaluations,
                            ),
                            evaluations
                        )
                    };
                    samples[first_point * second_stencil.len + second_point] = evaluated;
                }
            }
            let sample_count = first_stencil.len * second_stencil.len;
            // Cached copies invoke no provider. Poll before fallible contraction;
            // publication retains its final cancellation/deadline poll.
            if !evaluations.report.poll(evaluations.context)? {
                return Ok(evaluations.report);
            }
            let value = mixed_stencil_derivative(
                first_stencil,
                second_stencil,
                &samples[..sample_count],
                base,
                &mut workspace.reduction[0],
            )
            .map_err(|error| {
                differentiation_arithmetic_failure(
                    DifferentiationLocation::variables(first, second),
                    evaluations.report.evaluations(),
                    error,
                )
            })?;
            workspace.derivative[first * x.len() + second] = value;
            workspace.derivative[second * x.len() + first] = value;
            evaluations.report.completed_entries += 2;
        }
    }
    if !evaluations.report.poll(evaluations.context)? {
        return Ok(evaluations.report);
    }
    validate_then_commit(&workspace.derivative[..output.len()], output, |values| {
        validate_output(values, "Hessian output is non-finite")
    })?;
    Ok(evaluations.report)
}

// Positive, nonzero canonical components used by step selection and noise
// estimates. Arithmetic is checked; only representable_step treats final IEEE
// range loss as the legitimate absence of a usable step. Every fallible method
// reports failures selecting the step.
/// Computes a finite-difference Jacobian-vector product into `output`.
///
/// [`ZeroDirectionPolicy::MathematicalZero`] returns an exact zero JVP with
/// zero callback evaluations for an exactly zero direction.
/// [`ZeroDirectionPolicy::ValidateBase`] makes and validates exactly one base
/// callback before returning zero. For a nonzero direction, a missing base
/// value adds one evaluation and every attempted stencil callback—including a
/// failing callback—is counted once. The output remains unchanged if any error
/// is returned, and scratch remains reusable.
///
/// # Errors
///
/// Returns structured input, workspace, callback, non-finite, or numerical errors.
#[expect(
    clippy::too_many_arguments,
    reason = "numerical inputs, caller-owned staging and execution capabilities stay explicit"
)]
fn fixed_jvp<S, F>(
    options: &FiniteDifference<S>,
    zero_direction: ZeroDirectionPolicy,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    direction: &[S],
    base: Option<&[S]>,
    function: &mut F,
    output: &mut [S],
    workspace: &mut FixedBuffers<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<DifferentiationReport, SciOxideError>
where
    S: RealScalar,
    F: SampleFunction<S, S>,
{
    let scheme = options.scheme;
    let m = workspace.output_dimension;
    if output.is_empty() {
        let mut report =
            DifferentiationReport::fixed(DerivativeSource::FiniteDifference(scheme), output.len());
        report.poll(context)?;
        return Ok(report);
    }
    let Some(normalized) = normalized_direction(x, direction)? else {
        let mut evaluations = FixedEvaluation {
            context,
            policy: options.execution,
            report: DifferentiationReport::fixed(
                DerivativeSource::FiniteDifference(scheme),
                output.len(),
            ),
        };
        if zero_direction == ZeroDirectionPolicy::ValidateBase {
            admitted!(
                evaluate_vector(
                    function,
                    x,
                    &mut workspace.base_value,
                    DifferentiationLocation::direction(),
                    &mut evaluations,
                ),
                evaluations
            );
        }
        workspace.derivative[..m].fill(S::ZERO);
        evaluations.report.completed_entries = m;
        if !evaluations.report.poll(evaluations.context)? {
            return Ok(evaluations.report);
        }
        validate_then_commit(&workspace.derivative[..m], output, |values| {
            validate_output(values, "JVP output is non-finite")
        })?;
        return Ok(evaluations.report);
    };
    // `nominal_step` at zero returns the bare relative factor (sqrt or cbrt of
    // eps). Dividing by the normalized direction size chooses alpha so that the
    // fastest-moving coordinate is perturbed by the usual per-coordinate step.
    let nominal = if options.step.is_default() {
        PositiveMagnitude::from_positive(nominal_step(S::ZERO, scheme, 1))?.divide(normalized)?
    } else {
        let mut nominal = None;
        for (i, (&value, &component)) in x.iter().zip(direction).enumerate() {
            if component != S::ZERO {
                update_minimum(
                    &mut nominal,
                    Some(PositiveMagnitude::ratio(
                        options.nominal(value, i, 1),
                        component.abs(),
                    )?),
                );
            }
        }
        nominal.ok_or_else(|| step_size_limit_error(0, "nonzero direction required"))?
    };
    let alpha_bounds = directional_bounds(x, direction, bounds)?;
    let (point, remaining) = workspace.point.split_at_mut(x.len());
    let stencil = directional_stencil_for_points(
        nominal,
        scheme,
        alpha_bounds,
        x,
        direction,
        bounds,
        (point, &mut remaining[..x.len()]),
    )?
    .ok_or_else(|| step_size_limit_error(0, "no representable directional stencil is available"))?;
    for (sample, &alpha) in stencil.offsets[..stencil.len].iter().enumerate() {
        let point = &mut workspace.point[sample * x.len()..(sample + 1) * x.len()];
        if alpha == S::ZERO {
            point.copy_from_slice(x);
            continue;
        }
        if !generated_directional_point(x, direction, alpha, bounds, point)? {
            return Err(step_size_limit_error(
                0,
                "directional perturbation is unavailable at the active bounds",
            ));
        }
    }
    let mut evaluations = FixedEvaluation {
        context,
        policy: options.execution,
        report: DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(scheme),
            output.len(),
        ),
    };
    if !evaluations.report.iteration(evaluations.context)? {
        return Ok(evaluations.report);
    }
    admitted!(
        prepare_vector_base(
            function,
            x,
            base,
            &mut workspace.base_value,
            &mut evaluations,
            DifferentiationLocation::direction(),
        ),
        evaluations
    );
    for point_index in 0..stencil.len {
        let alpha = stencil.offsets[point_index];
        if alpha == S::ZERO {
            workspace.value.copy_from_slice(&workspace.base_value);
        } else {
            admitted!(
                evaluate_vector(
                    function,
                    &workspace.point[point_index * x.len()..(point_index + 1) * x.len()],
                    &mut workspace.value,
                    DifferentiationLocation::direction(),
                    &mut evaluations
                ),
                evaluations
            );
        }
        for component in 0..m {
            workspace.stencil_values[component * MAX_STENCIL_POINTS + point_index] =
                workspace.value[component];
        }
    }
    for component in 0..m {
        let start = component * MAX_STENCIL_POINTS;
        workspace.derivative[component] = stencil_derivative(
            stencil,
            &workspace.stencil_values[start..start + stencil.len],
            workspace.base_value[component],
            &mut workspace.reduction[0],
        )
        .map_err(|error| {
            differentiation_arithmetic_failure(
                DifferentiationLocation::direction().with_component(component),
                evaluations.report.evaluations(),
                error,
            )
        })?;
        evaluations.report.completed_entries += 1;
    }
    if !evaluations.report.poll(evaluations.context)? {
        return Ok(evaluations.report);
    }
    validate_then_commit(&workspace.derivative[..m], output, |values| {
        validate_output(values, "JVP output is non-finite")
    })?;
    Ok(evaluations.report)
}

#[cfg(test)]
mod tests;
