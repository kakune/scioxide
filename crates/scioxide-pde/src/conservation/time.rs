use super::plan::ConservationCore;
use super::{
    ConservationLaw, ConservationReport, ConservationWorkspace, NumericalFlux,
    SimplexConservationPlan, StructuredConservationPlan, invalid,
};
use core::cell::RefCell;
use scioxide_core::{EvaluationRole, ExecutionContext, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;
use scioxide_ode::{OdeRhsStepLimit, OdeSystem};
use scioxide_spatial::{Cell, CellField, FieldView};

/// A borrowed, mass-normalized nonlinear ODE operator over cell-major conserved values.
/// The immutable plan and providers are shared; numerical scratch is exclusively borrowed.
/// No Jacobian or dense/sparse linearization is advertised.
///
/// The system enforces admissibility itself, for every ODE method: through
/// [`OdeSystem::admits`] it declares inadmissible stage inputs and candidates, so the
/// plan discards the trial without modifying the accepted state and retries with half
/// the step, stopping with `StepSizeLimit` at the configured bound. A direct `rhs` call
/// at an inadmissible state is an error (argument `y`). Nothing is clipped.
/// Explicit Runge–Kutta plans obtain the CFL bound with their first residual through
/// [`OdeSystem::rhs_with_step_limit`]; one residual is evaluated per stage. Other
/// methods obtain the same bound through `step_limit`. Both paths use the caller's
/// execution context and the conservation options' execution policy.
///
/// Run the built-in strong-stability-preserving [`scioxide_ode::ForwardEuler`] or
/// [`scioxide_ode::Heun`] (SSPRK2) through [`scioxide_ode::OdeMethod`]. Heun's first
/// forward-Euler subupdate is its second stage input; its second subupdate is covered
/// through the convex combination checked as the candidate.
///
/// ```
/// use scioxide_pde::{ConservationSystem, Godunov, LinearAdvection, StructuredConservationPlan};
/// use scioxide_spatial::{AxisTopology, CellField, CoordinateAxis, Grid1d, SegmentMesh};
/// use scioxide_ode::{Heun, OdeMethod, OdeOptions, OdeProblem, OdeState};
/// let mesh = SegmentMesh::new(Grid1d::new([CoordinateAxis::uniform(0.0, 1.0, 9)?], [AxisTopology::Periodic])?)?;
/// let plan = StructuredConservationPlan::<f64, 1, 1>::new(mesh, Vec::new())?;
/// let law = LinearAdvection { velocity: [1.0] };
/// let field = CellField::from_fn(plan.mesh(), |_| [1.0])?;
/// let mut workspace = plan.workspace()?;
/// let system = plan.bind(&law, &Godunov, &mut workspace)?;
/// let initial = OdeState::new(0.0, plan.initial_state(&law, field.view())?);
/// let result = Heun::new().solve(&OdeProblem::new(&system, 0.5),
///     &OdeOptions::new().with_initial_step(0.05), initial)?;
/// assert!(result.state().values().iter().all(|u| (u - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct ConservationSystem<'a, R: RealScalar, E, F, const D: usize, const N: usize> {
    core: &'a ConservationCore<R, D, N>,
    equation: &'a E,
    flux: &'a F,
    workspace: RefCell<&'a mut ConservationWorkspace<R, D, N>>,
}
impl<
    R: RealScalar,
    E: ConservationLaw<R, D, N>,
    F: NumericalFlux<R, E, D, N>,
    const D: usize,
    const N: usize,
> ConservationSystem<'_, R, E, F, D, N>
{
    fn evaluate(
        &self,
        time: R,
        state: &[R],
        output: Option<&mut [R]>,
        execution: ExecutionContext<'_>,
    ) -> Result<ConservationReport<R, N>, SciOxideError> {
        const SELF: scioxide_core::Argument = scioxide_core::argument!("self");
        if state.len() != self.dimension()
            || output.as_ref().is_some_and(|o| o.len() != self.dimension())
        {
            return Err(invalid(
                if state.len() == self.dimension() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("state")
                },
                "conservation ODE state/output shape mismatch",
            ));
        }
        let (cells, remainder) = state.as_chunks::<N>();
        if !remainder.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("state"),
                "incomplete conserved state",
            ));
        }
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| invalid(SELF, "conservation workspace already borrowed"))?;
        let report = self.core.evaluate(
            self.equation,
            self.flux,
            time,
            cells,
            &mut workspace,
            execution,
            SELF,
        )?;
        if let Some(out) = output {
            out.copy_from_slice(workspace.residual.as_flattened());
        }
        Ok(report)
    }
    /// Inspect CFL, boundary balance and limiter diagnostics without retaining a cache.
    /// # Errors
    /// Propagates shape, callback, admissibility and numerical failures.
    pub fn report(
        &self,
        time: R,
        state: &[R],
        execution: ExecutionContext<'_>,
    ) -> Result<ConservationReport<R, N>, SciOxideError> {
        self.evaluate(time, state, None, execution)
    }
}
impl<
    R: RealScalar,
    E: ConservationLaw<R, D, N>,
    F: NumericalFlux<R, E, D, N>,
    const D: usize,
    const N: usize,
> OdeSystem<R> for ConservationSystem<'_, R, E, F, D, N>
{
    fn dimension(&self) -> usize {
        self.core.geometry.centers.len() * N
    }
    /// Reject an inadmissible stage input (argument `y`) before evaluating the residual.
    fn rhs(
        &self,
        time: R,
        state: &[R],
        output: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        admissible(self.equation, state, scioxide_core::argument!("y"))?;
        self.evaluate(time, state, Some(output), context.execution())
            .map(|_| ())
    }
    /// Inadmissible stage inputs and candidates reject the trial; the plan retries with
    /// half the step.
    fn admits(
        &self,
        _: R,
        state: &[R],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        let (cells, tail) = state.as_chunks::<N>();
        Ok(tail.is_empty() && cells.iter().all(|u| self.equation.admissible(u)))
    }
    fn rhs_with_step_limit(
        &self,
        time: R,
        state: &[R],
        output: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<OdeRhsStepLimit<R>, SciOxideError> {
        context.execution().check()?;
        context
            .record_provider(EvaluationRole::RightHandSide)?
            .require()?;
        let report = self.evaluate(time, state, Some(output), context.execution())?;
        Ok(OdeRhsStepLimit::Evaluated {
            maximum_step: report.maximum_step,
        })
    }
    fn step_limit(
        &self,
        time: R,
        state: &[R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<R>, SciOxideError> {
        context.execution().check()?;
        if !context
            .record_provider(EvaluationRole::Function)?
            .is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::EvaluationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                "conservation CFL evaluation budget exhausted",
            ));
        }
        self.evaluate(time, state, None, context.execution())
            .map(|r| r.maximum_step)
    }
}

/// Reject inadmissible states; `argument` names the caller's state.
fn admissible<R: RealScalar, E: ConservationLaw<R, D, N>, const D: usize, const N: usize>(
    e: &E,
    state: &[R],
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    let (cells, tail) = state.as_chunks::<N>();
    if !tail.is_empty() || cells.iter().any(|u| !e.admissible(u)) {
        return Err(invalid(argument, "inadmissible conservation time state"));
    }
    Ok(())
}
macro_rules! bind_plan {
    ($plan:ident) => {
        impl<R: RealScalar, const D: usize, const N: usize> $plan<R, D, N> {
            /// Borrow a spatial workspace and providers as the existing ODE system protocol.
            /// # Errors
            /// Rejects an invalid equation or incompatible workspace before callbacks.
            pub fn bind<'a, E: ConservationLaw<R, D, N>, F: NumericalFlux<R, E, D, N>>(
                &'a self,
                equation: &'a E,
                flux: &'a F,
                workspace: &'a mut ConservationWorkspace<R, D, N>,
            ) -> Result<ConservationSystem<'a, R, E, F, D, N>, SciOxideError> {
                equation.validate()?;
                if workspace.residual.len() != self.volumes().len() {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        "conservation bind workspace mismatch",
                    ));
                }
                Ok(ConservationSystem {
                    core: &self.core,
                    equation,
                    flux,
                    workspace: RefCell::new(workspace),
                })
            }
            /// Validate and flatten a typed initial field into an owned ODE vector.
            /// # Errors
            /// Rejects mesh mismatch, inadmissible data and allocation failure.
            pub fn initial_state<E: ConservationLaw<R, D, N>>(
                &self,
                e: &E,
                field: FieldView<'_, [R; N], Cell, D>,
            ) -> Result<Vector<R>, SciOxideError> {
                self.core.token.ensure_compatible(field.mesh_token())?;
                e.validate()?;
                admissible(
                    e,
                    field.values().as_flattened(),
                    scioxide_core::argument!("field"),
                )?;
                let mut result = Vector::zeros(self.unknowns())?;
                result
                    .as_mut_slice()
                    .copy_from_slice(field.values().as_flattened());
                Ok(result)
            }
            /// Decode an accepted ODE state into a typed owned field, without another mass scaling.
            /// # Errors
            /// Rejects wrong shape, inadmissibility and allocation failure.
            pub fn field<E: ConservationLaw<R, D, N>>(
                &self,
                e: &E,
                state: &[R],
            ) -> Result<CellField<[R; N], D>, SciOxideError> {
                if state.len() != self.unknowns() {
                    return Err(invalid(
                        scioxide_core::argument!("state"),
                        "conservation field shape mismatch",
                    ));
                }
                e.validate()?;
                admissible(e, state, scioxide_core::argument!("state"))?;
                FieldView::<_, Cell, D>::new(self.mesh(), state.as_chunks::<N>().0)?.to_owned()
            }
        }
    };
}
bind_plan!(StructuredConservationPlan);
bind_plan!(SimplexConservationPlan);
