use super::*;
use scioxide_core::SolveScope;

/// Plan-fixed staging and operator scratch of [`PdeTransferDerivatives`] products.
#[derive(Debug, Clone)]
pub struct PdeTransferWorkspace<T> {
    work: Vec<T>,
}
/// Derivatives of an explicit field transfer at a fixed input and selected topology.
///
/// The state operator is the actual transfer derivative (including constraints).
/// An optional real-parameter derivative operator carries parameter dependence of
/// the transfer (for example supplied smooth weight derivatives); without it the
/// transfer is parameter independent. This also composes supplied
/// parameter-dependent initial/reconstruction maps.
pub struct PdeTransferDerivatives<'a, T: NumericScalar> {
    source: &'a MeshToken,
    destination: &'a MeshToken,
    state: &'a dyn AdjointOperator<T>,
    parameters: Option<&'a dyn AdjointOperator<T>>,
    parameter_count: usize,
    scratch: usize,
}
impl<'a, T: NumericScalar> PdeTransferDerivatives<'a, T> {
    /// Prepare exact epoch checks and scratch requirements without allocating.
    /// # Errors
    /// Rejects a parameter operator whose shape differs from the state rows and
    /// `parameter_count` columns.
    pub fn new(
        source: &'a MeshToken,
        destination: &'a MeshToken,
        state: &'a dyn AdjointOperator<T>,
        parameters: Option<&'a dyn AdjointOperator<T>>,
        parameter_count: usize,
    ) -> Result<Self, SciOxideError> {
        if parameters.is_some_and(|p| p.rows() != state.rows() || p.cols() != parameter_count) {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                "transfer parameter derivatives are incompatible",
            ));
        }
        let scratch = state
            .apply_workspace_len()?
            .max(state.apply_adjoint_workspace_len()?)
            .max(match parameters {
                Some(p) => p
                    .apply_workspace_len()?
                    .max(p.apply_adjoint_workspace_len()?),
                None => 0,
            });
        Ok(Self {
            source,
            destination,
            state,
            parameters,
            parameter_count,
            scratch,
        })
    }
    /// Exact scalar scratch including transactional staging and parameter conversion.
    /// # Errors
    /// Returns checked size overflow.
    fn staging_len(&self) -> Result<usize, SciOxideError> {
        const WORKSPACE: scioxide_core::Quantity = scioxide_core::quantity!("transfer_staging");
        checked_add(
            checked_add(
                checked_mul(self.state.rows(), 2, WORKSPACE)?,
                self.state.cols(),
                WORKSPACE,
            )?,
            checked_add(self.parameter_count, self.scratch, WORKSPACE)?,
            WORKSPACE,
        )
    }

    /// Staging and operator scratch of both products; it never grows.
    /// # Errors
    /// Returns checked size overflow and allocation failures.
    pub fn workspace(&self) -> Result<PdeTransferWorkspace<T>, SciOxideError> {
        Ok(PdeTransferWorkspace {
            work: try_filled(
                self.staging_len()?,
                T::ZERO,
                scioxide_core::quantity!("transfer_staging"),
            )?,
        })
    }

    fn check(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        work: &[T],
    ) -> Result<(), SciOxideError> {
        self.source.ensure_compatible(source)?;
        self.destination.ensure_compatible(destination)?;
        if work.len() < self.staging_len()?
            || self.state.apply_workspace_len()? > self.scratch
            || self.state.apply_adjoint_workspace_len()? > self.scratch
        {
            return Err(invalid(
                scioxide_core::argument!("work"),
                "transfer scratch changed or is insufficient",
            ));
        }
        if let Some(p) = self.parameters
            && (p.rows() != self.state.rows()
                || p.cols() != self.parameter_count
                || p.apply_workspace_len()? > self.scratch
                || p.apply_adjoint_workspace_len()? > self.scratch)
        {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "transfer parameter shape/scratch changed",
            ));
        }
        Ok(())
    }
    /// Apply the full transfer JVP; parameter directions are real.
    /// # Errors
    /// Invalid epochs, products, budgets and nonfinite values preserve output.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub fn jvp_into<'scope, 'capability>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        state_direction: &[T],
        parameter_direction: &[T::Real],
        output: &mut [T],
        workspace: &mut PdeTransferWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "transfer-jvp", 2),
            |context| {
                self.jvp(
                    source,
                    destination,
                    state_direction,
                    parameter_direction,
                    output,
                    &mut workspace.work,
                    context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    fn jvp(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        state_direction: &[T],
        parameter_direction: &[T::Real],
        output: &mut [T],
        work: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.check(source, destination, work)?;
        if state_direction.len() != self.state.cols()
            || output.len() != self.state.rows()
            || parameter_direction.len() != self.parameter_count
        {
            return Err(invalid(
                if state_direction.len() != self.state.cols() {
                    scioxide_core::argument!("state_direction")
                } else if output.len() != self.state.rows() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("parameter_direction")
                },
                "transfer JVP shape",
            ));
        }
        finite_slice(state_direction, scioxide_core::NonFiniteOrigin::Input)?;
        finite_slice(parameter_direction, scioxide_core::NonFiniteOrigin::Input)?;
        let (staged, rest) = work.split_at_mut(self.state.rows());
        let (extra, rest) = rest.split_at_mut(self.state.rows());
        let (parameters, scratch) = rest.split_at_mut(self.parameter_count);
        poison(staged);
        admit(context, EvaluationRole::LinearOperator)?;
        self.state
            .apply_into_scoped(
                state_direction,
                staged,
                &mut scratch[..self.state.apply_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(staged, scioxide_core::NonFiniteOrigin::Callback)?;
        if let Some(p) = self.parameters {
            for (out, &v) in parameters.iter_mut().zip(parameter_direction) {
                *out = T::from_real(v);
            }
            poison(extra);
            admit(context, EvaluationRole::LinearOperator)?;
            p.apply_into_scoped(
                parameters,
                extra,
                &mut scratch[..p.apply_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
            finite_slice(extra, scioxide_core::NonFiniteOrigin::Callback)?;
            for (out, &v) in staged.iter_mut().zip(extra.iter()) {
                *out = *out + v;
            }
        }
        finite_slice(staged, scioxide_core::NonFiniteOrigin::Callback)?;
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
    /// Reverse transfer with the source/destination epochs explicitly checked.
    /// Both cotangents are committed together; the parameter result is a contribution
    /// to add to adjacent epoch gradients, not an implicit global accumulation.
    /// # Errors
    /// Provider, resource and nonfinite failures preserve both output buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub fn pullback_into<'scope, 'capability>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[T],
        state_output: &mut [T],
        parameter_output: &mut [T::Real],
        workspace: &mut PdeTransferWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "transfer-pullback", 2),
            |context| {
                self.pullback(
                    source,
                    destination,
                    seed,
                    state_output,
                    parameter_output,
                    &mut workspace.work,
                    context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    fn pullback(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[T],
        state_output: &mut [T],
        parameter_output: &mut [T::Real],
        work: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.check(source, destination, work)?;
        if seed.len() != self.state.rows()
            || state_output.len() != self.state.cols()
            || parameter_output.len() != self.parameter_count
        {
            return Err(invalid(
                if seed.len() != self.state.rows() {
                    scioxide_core::argument!("seed")
                } else if state_output.len() != self.state.cols() {
                    scioxide_core::argument!("state_output")
                } else {
                    scioxide_core::argument!("parameter_output")
                },
                "transfer pullback shape",
            ));
        }
        finite_slice(seed, scioxide_core::NonFiniteOrigin::Input)?;
        let (staged, rest) = work.split_at_mut(self.state.cols());
        let (parameters, scratch) = rest.split_at_mut(self.parameter_count);
        poison(staged);
        admit(context, EvaluationRole::LinearOperator)?;
        self.state
            .apply_adjoint_into_scoped(
                seed,
                staged,
                &mut scratch[..self.state.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(staged, scioxide_core::NonFiniteOrigin::Callback)?;
        parameters.fill(T::ZERO);
        if let Some(p) = self.parameters {
            poison(parameters);
            admit(context, EvaluationRole::LinearOperator)?;
            p.apply_adjoint_into_scoped(
                seed,
                parameters,
                &mut scratch[..p.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
            finite_slice(parameters, scioxide_core::NonFiniteOrigin::Callback)?;
        }
        context.execution().check()?;
        state_output.copy_from_slice(staged);
        for (out, &v) in parameter_output.iter_mut().zip(parameters.iter()) {
            *out = v.real_component();
        }
        Ok(())
    }
}
