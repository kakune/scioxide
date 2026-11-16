//! Transforms along one axis of an N-D array for the chirp-z, nonuniform,
//! discrete Hankel and FFTLog plans.
//!
//! Each plan implements [`AxisPlan`] for one lane; [`requirements`] and
//! [`execute`] handle axis sizing, byte limits, workspace checks, lane staging
//! and the final write to the output.
use crate::validation::length;
use crate::workspace::nested_byte_limit;
use crate::{AxisWorkspaceRequest, FftWorkspace, FftWorkspaceRequirements};
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    Budget, BudgetKind, Complex, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveScope,
};

/// A prepared plan applied independently to every lane of a selected axis.
pub(crate) trait AxisPlan<S: RealScalar> {
    /// Input and output lane lengths.
    fn lane_lens(&self) -> (usize, usize);
    /// Plain one-transform storage, with child limits already composed. It
    /// includes the plan's own candidate output.
    fn scratch(&self) -> Result<FftWorkspaceRequirements, SciOxideError>;
    /// Configured cap on all workspace bytes.
    fn workspace_bytes(&self) -> usize;
    /// Configured iteration allowance of each solve epoch.
    fn iteration_budget(&self) -> Option<Budget> {
        None
    }
    /// Budgets of a workspace-owned or operator solve epoch.
    fn budgets(&self) -> SolveBudgets {
        let budgets = SolveBudgets::unbounded();
        self.iteration_budget().map_or(budgets, |budget| {
            budgets.with_limit(BudgetKind::Iteration, budget)
        })
    }
    /// Check and charge a nonempty axis call once, before its first lane.
    fn admit(&self, _context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        Ok(())
    }
    /// Transform one finite lane, leaving its results inside `scratch`.
    fn lane<'w>(
        &self,
        source: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError>;
}

struct Geometry {
    batches: usize,
    inner: usize,
    outputs: usize,
    required: usize,
}

impl Geometry {
    fn new(
        shape: &[usize],
        axis: usize,
        (n, m): (usize, usize),
        scratch: usize,
    ) -> Result<Self, SciOxideError> {
        let extent = scioxide_core::Shape::new(shape)?.extent(axis)?;
        length(extent, n)?;
        // A zero transform extent can still describe nonempty batches of empty
        // NUFFT sums. Only a zero complementary extent suppresses all lanes.
        if shape.iter().enumerate().any(|(i, &n)| i != axis && n == 0) {
            return Ok(Self {
                batches: 0,
                inner: 0,
                outputs: 0,
                required: 0,
            });
        }
        let product = |xs: &[usize]| {
            xs.iter().try_fold(1, |a, &b| {
                checked_mul(a, b, scioxide_core::quantity!("axis_batches"))
            })
        };
        let inner = product(&shape[axis + 1..])?;
        let batches = checked_mul(
            product(&shape[..axis])?,
            inner,
            scioxide_core::quantity!("axis_batches"),
        )?;
        let outputs = checked_mul(batches, m, scioxide_core::quantity!("axis_outputs"))?;
        // Each lane's results stay in the plan's own transactional candidate.
        let required = checked_add(
            outputs,
            checked_add(n, scratch, scioxide_core::quantity!("axis_scratch"))?,
            scioxide_core::quantity!("axis_scratch"),
        )?;
        Ok(Self {
            batches,
            inner,
            outputs,
            required,
        })
    }

    /// Whole-transaction storage; the plain scratch keeps its own limit.
    fn requirements<S: RealScalar>(
        &self,
        plan: &impl AxisPlan<S>,
        scratch: FftWorkspaceRequirements,
    ) -> Result<FftWorkspaceRequirements, SciOxideError> {
        let staged = FftWorkspaceRequirements::new::<S>(self.required)?;
        staged.with_limit(nested_byte_limit(
            plan.workspace_bytes(),
            staged.bytes(),
            [(scratch.byte_limit(), scratch.bytes())],
        ))
    }
}

/// Describe plain storage or one whole selected-axis transaction.
pub(crate) fn requirements<S: RealScalar>(
    plan: &impl AxisPlan<S>,
    request: AxisWorkspaceRequest<'_>,
) -> Result<FftWorkspaceRequirements, SciOxideError> {
    let scratch = plan.scratch()?;
    match request {
        AxisWorkspaceRequest::Plain => Ok(scratch),
        AxisWorkspaceRequest::Axis { shape, axis } => {
            Geometry::new(shape, axis, plan.lane_lens(), scratch.complex_len())?
                .requirements(plan, scratch)
        }
    }
}

/// Transform every lane of `axis` in one workspace-owned epoch, writing the
/// output only after all lanes succeed.
pub(crate) fn execute<S: RealScalar, T: NumericScalar<Real = S>>(
    plan: &impl AxisPlan<S>,
    input: NdView<'_, '_, T>,
    axis: usize,
    mut output: NdViewMut<'_, '_, Complex<S>>,
    workspace: &mut FftWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<(), SciOxideError> {
    let shape = input.layout().shape().extents();
    let (n, m) = plan.lane_lens();
    let scratch = plan.scratch()?;
    let geometry = Geometry::new(shape, axis, (n, m), scratch.complex_len())?;
    let requirements = geometry.requirements(plan, scratch)?;
    let out_shape = output.layout().shape();
    length(out_shape.rank(), shape.len())?;
    for (i, (&actual, &original)) in out_shape.extents().iter().zip(shape).enumerate() {
        length(actual, if i == axis { m } else { original })?;
    }
    workspace.run(
        &requirements,
        n,
        plan.budgets(),
        scope,
        |values, context| {
            if geometry.batches == 0 {
                return Ok(());
            }
            plan.admit(context)?;
            let (candidate, rest) = values.split_at_mut(geometry.outputs);
            let (source, work) = rest.split_at_mut(n);
            let lanes = input.layout().axis(axis)?;
            for batch in 0..geometry.batches {
                for (out, offset) in source.iter_mut().zip(lanes.lane(batch)?.offsets()) {
                    let value = input.storage()[offset];
                    if !value.has_finite_components() {
                        return Err(crate::validation::nonfinite());
                    }
                    *out = Complex::new(value.real_component(), value.imaginary_component());
                }
                let result = plan.lane(source, work, context)?;
                let (outer, lane) = (batch / geometry.inner, batch % geometry.inner);
                for (k, &value) in result.iter().enumerate() {
                    candidate[(outer * m + k) * geometry.inner + lane] = value;
                }
            }
            context.execution().check()?;
            let mut values = candidate.iter().copied();
            output.for_each_mut(|out| {
                if let Some(value) = values.next() {
                    *out = value;
                }
            });
            Ok(())
        },
    )
}

/// A radial operator whose forward and adjoint results stay in caller scratch.
pub(crate) trait RadialPlan<S: RealScalar>: AxisPlan<S> {
    fn evaluate<'w>(
        &self,
        input: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        adjoint: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError>;
}

/// Execute a radial vector in the reusable workspace's solve epoch.
pub(crate) fn execute_radial<S: RealScalar>(
    plan: &impl RadialPlan<S>,
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    workspace: &mut FftWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<(), SciOxideError> {
    workspace.run(
        &plan.scratch()?,
        plan.lane_lens().0,
        plan.budgets(),
        scope,
        |scratch, context| publish_radial(plan, input, output, scratch, false, context),
    )
}

/// Execute a borrowed radial operator in one root or child solve epoch.
pub(crate) fn apply_radial<S: RealScalar>(
    plan: &impl RadialPlan<S>,
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    scratch: &mut [Complex<S>],
    adjoint: bool,
    scope: SolveScope<'_, '_>,
) -> Result<(), SciOxideError> {
    scope.run(
        &mut scioxide_core::SolveState::new(),
        crate::workspace::identity(plan.lane_lens().0),
        plan.budgets(),
        |context| publish_radial(plan, input, output, scratch, adjoint, context),
    )
}

fn publish_radial<S: RealScalar>(
    plan: &impl RadialPlan<S>,
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    scratch: &mut [Complex<S>],
    adjoint: bool,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    length(output.len(), plan.lane_lens().1)?;
    length(input.len(), plan.lane_lens().0)?;
    crate::validation::sufficient(scratch.len(), plan.scratch()?.complex_len())?;
    plan.admit(context)?;
    output.copy_from_slice(plan.evaluate(input, scratch, adjoint, context)?);
    Ok(())
}

// Foreign operator traits require concrete plan impls. Their identical
// forwarding methods share lifecycle and publication through the driver.
macro_rules! impl_radial_operators {
    ($plan:ident) => {
        impl<S: scioxide_core::RealScalar> scioxide_core::LinearOperator<scioxide_core::Complex<S>>
            for $plan<S>
        where
            scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
        {
            fn rows(&self) -> usize {
                crate::axis::AxisPlan::lane_lens(self).1
            }
            fn cols(&self) -> usize {
                crate::axis::AxisPlan::lane_lens(self).0
            }
            fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(crate::axis::AxisPlan::scratch(self)?.complex_len())
            }
            fn apply_into(
                &self,
                input: &[scioxide_core::Complex<S>],
                output: &mut [scioxide_core::Complex<S>],
                workspace: &mut [scioxide_core::Complex<S>],
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::axis::apply_radial(
                    self,
                    input,
                    output,
                    workspace,
                    false,
                    scioxide_core::ExecutionContext::sequential_native().into(),
                )
            }
            fn apply_into_scoped(
                &self,
                input: &[scioxide_core::Complex<S>],
                output: &mut [scioxide_core::Complex<S>],
                workspace: &mut [scioxide_core::Complex<S>],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, scioxide_core::Complex<S>>,
            ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
                crate::axis::apply_radial(
                    self,
                    input,
                    output,
                    workspace,
                    false,
                    (&mut **context).into(),
                )
                .map(scioxide_core::Invocation::Evaluated)
            }
        }
        impl<S: scioxide_core::RealScalar> scioxide_core::AdjointOperator<scioxide_core::Complex<S>>
            for $plan<S>
        where
            scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
        {
            fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(crate::axis::AxisPlan::scratch(self)?.complex_len())
            }
            fn apply_adjoint_into(
                &self,
                input: &[scioxide_core::Complex<S>],
                output: &mut [scioxide_core::Complex<S>],
                workspace: &mut [scioxide_core::Complex<S>],
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::axis::apply_radial(
                    self,
                    input,
                    output,
                    workspace,
                    true,
                    scioxide_core::ExecutionContext::sequential_native().into(),
                )
            }
            fn apply_adjoint_into_scoped(
                &self,
                input: &[scioxide_core::Complex<S>],
                output: &mut [scioxide_core::Complex<S>],
                workspace: &mut [scioxide_core::Complex<S>],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, scioxide_core::Complex<S>>,
            ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
                crate::axis::apply_radial(
                    self,
                    input,
                    output,
                    workspace,
                    true,
                    (&mut **context).into(),
                )
                .map(scioxide_core::Invocation::Evaluated)
            }
        }
    };
}
pub(crate) use impl_radial_operators;
