//! Reusable storage for explicit, implicit, and multistep ODE solvers.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

use crate::stepper::StepRequest;
use crate::stiff::kernel::{CacheStats, KernelShape, StageKernel};

use crate::dense_output::DenseOutputStorage;

/// Accepted-point validity for a derivative held in separate prepared storage.
#[derive(Debug, Clone, Default)]
pub(crate) struct DerivativeCache {
    current_step: Option<usize>,
}
impl DerivativeCache {
    pub(crate) fn reset(&mut self) {
        self.current_step = None;
    }
    /// Record that the derivative storage holds `f` at this accepted base.
    pub(crate) fn hold(&mut self, accepted_step: Option<usize>) {
        self.current_step = accepted_step;
    }
    pub(crate) fn advance(&mut self) {
        self.current_step = self.current_step.and_then(|step| step.checked_add(1));
    }
    pub(crate) fn contains(&self, accepted_step: Option<usize>) -> bool {
        accepted_step.is_some() && self.current_step == accepted_step
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_into<S, T, System>(
        &mut self,
        system: &System,
        time: S,
        state: &Vector<T>,
        request: StepRequest<'_, S, T>,
        derivative: &mut Vector<T>,
        candidate: &mut Vector<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        T: NumericScalar<Real = S>,
        System: crate::OdeSystem<S, T> + ?Sized,
    {
        if self.contains(request.accepted_step) {
            return Ok(());
        }
        self.reset();
        // The driver hands over a base derivative it has already admitted.
        if let Some(initial) = request.initial_rhs {
            derivative.as_mut_slice().copy_from_slice(initial);
        } else {
            crate::traits::evaluate_rhs_at_base(
                system, time, state, derivative, candidate, request, cx,
            )?;
        }
        self.current_step = request.accepted_step;
        Ok(())
    }
}

/// Reusable stage and trial storage for explicit ODE methods.
///
/// A workspace may grow on first use or after a dimension change. Once sized
/// for a method and dimension, steps reuse its storage without allocating. The
/// driver identifies accepted base points within one solve epoch for derivative reuse.
/// Acceptance keeps an available endpoint derivative; rejection keeps the
/// starting derivative. A fresh epoch invalidates this cache.
#[derive(Debug, Clone)]
#[must_use]
pub(crate) struct ExplicitOdeWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    pub(crate) stages: Vec<Vector<T>>,
    pub(crate) work: Vector<T>,
    pub(crate) trial: Vector<T>,
    pub(crate) error: Vector<T>,
    pub(crate) endpoint_rhs: Vector<T>,
    pub(crate) rhs_candidate: Vector<T>,
    pub(crate) cache: DerivativeCache,
    pub(crate) pending_endpoint: bool,
    pub(crate) dense_output: DenseOutputStorage<S, T>,
    dimension: usize,
    prepared: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>
    ExplicitOdeWorkspace<S, T>
{
    pub(crate) fn reset_epoch(&mut self) {
        self.cache.reset();
        self.pending_endpoint = false;
    }
    pub(crate) fn accept_step(&mut self) {
        if core::mem::take(&mut self.pending_endpoint) {
            core::mem::swap(&mut self.stages[0], &mut self.endpoint_rhs);
            self.cache.advance();
        } else {
            self.cache.reset();
        }
    }
    pub(crate) fn discard_step(&mut self) {
        self.pending_endpoint = false;
    }
    /// Create an empty workspace without allocating.
    pub(crate) const fn new() -> Self {
        Self {
            stages: Vec::new(),
            work: Vector::empty(),
            trial: Vector::empty(),
            error: Vector::empty(),
            endpoint_rhs: Vector::empty(),
            rhs_candidate: Vector::empty(),
            cache: DerivativeCache { current_step: None },
            pending_endpoint: false,
            dense_output: DenseOutputStorage::new(),
            dimension: 0,
            prepared: false,
        }
    }

    /// Whether `try_resize(dimension, stages, dense_degree)` would keep this storage unchanged.
    pub(crate) fn fits(&self, dimension: usize, stages: usize, dense_degree: usize) -> bool {
        self.prepared
            && self.dimension == dimension
            && self.stages.len() >= stages
            && self.dense_output.fits(dimension, dense_degree)
    }

    /// Reserve stage vectors and continuous-extension coefficients, keeping capacity.
    /// Changing the dimension invalidates the derivative cache.
    /// # Errors
    /// Returns checked size or allocation failures.
    pub(crate) fn try_resize(
        &mut self,
        dimension: usize,
        stages: usize,
        dense_degree: usize,
    ) -> Result<(), SciOxideError> {
        if self.fits(dimension, stages, dense_degree) {
            return Ok(());
        }
        self.prepared = false;
        self.reset_epoch();
        let stage_values = checked_mul(
            dimension,
            stages,
            scioxide_core::quantity!("explicit_ode_stage_storage"),
        )?;
        let vector_values = checked_mul(
            dimension,
            5,
            scioxide_core::quantity!("explicit_ode_vector_storage"),
        )?;
        checked_add(
            stage_values,
            vector_values,
            scioxide_core::quantity!("explicit_ode_workspace_storage"),
        )?;

        if self.stages.len() < stages {
            let additional = stages - self.stages.len();
            try_reserve(
                &mut self.stages,
                additional,
                scioxide_core::quantity!("explicit_ode_stage_descriptors"),
            )?;
            self.stages.resize_with(stages, Vector::<T>::empty);
        }
        for stage in &mut self.stages {
            stage.try_resize(dimension, T::ZERO)?;
        }
        for vector in [
            &mut self.work,
            &mut self.trial,
            &mut self.error,
            &mut self.endpoint_rhs,
            &mut self.rhs_candidate,
        ] {
            vector.try_resize(dimension, T::ZERO)?;
        }
        self.dense_output.try_resize(dimension, dense_degree)?;

        self.dimension = dimension;
        self.prepared = true;
        Ok(())
    }

    /// Number of allocated stage-vector descriptors.
    #[must_use]
    pub(crate) const fn stage_capacity(&self) -> usize {
        self.stages.len()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for ExplicitOdeWorkspace<S, T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Reusable stage storage and the stiff kernel of implicit ODE methods.
///
/// A workspace may grow after a dimension, stage-count or backend change. Once sized,
/// prepared implicit steps reuse all stage vectors, Jacobians, finite-difference
/// scratch, factors and correction buffers without heap allocation.
#[derive(Debug)]
#[must_use]
pub(crate) struct ImplicitOdeWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    pub(crate) radau: Option<crate::implicit_rk::radau_iia::RadauWorkspace<S, T>>,
    pub(crate) kernel: Option<StageKernel<S, T>>,
    pub(crate) stages: Vec<Vector<T>>,
    pub(crate) stage_rhs: Vec<Vector<T>>,
    pub(crate) current_rhs: Vector<T>,
    pub(crate) endpoint_rhs: Vector<T>,
    pub(crate) cache: DerivativeCache,
    pub(crate) pending_endpoint: bool,
    pub(crate) coupled_state: Vector<T>,
    pub(crate) trial: Vector<T>,
    pub(crate) work: Vector<T>,
    pub(crate) weighted_rhs: Vector<T>,
    pub(crate) residual: Vector<T>,
    pub(crate) correction: Vector<T>,
    pub(crate) rhs_candidate: Vector<T>,
    dimension: usize,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>
    ImplicitOdeWorkspace<S, T>
{
    /// Create an empty implicit workspace without allocating.
    pub(crate) const fn new() -> Self {
        Self {
            radau: None,
            kernel: None,
            stages: Vec::new(),
            stage_rhs: Vec::new(),
            current_rhs: Vector::empty(),
            endpoint_rhs: Vector::empty(),
            cache: DerivativeCache { current_step: None },
            pending_endpoint: false,
            coupled_state: Vector::empty(),
            trial: Vector::empty(),
            work: Vector::empty(),
            weighted_rhs: Vector::empty(),
            residual: Vector::empty(),
            correction: Vector::empty(),
            rhs_candidate: Vector::empty(),
            dimension: 0,
        }
    }

    /// Whether `prepare(dimension, shape)` would keep this storage unchanged.
    pub(crate) fn fits(&self, dimension: usize, shape: KernelShape<'_>) -> bool {
        self.dimension == dimension
            && self.stages.len() == shape.stages
            && self.endpoint_rhs.len() == dimension
            && dimension
                .checked_mul(shape.stages)
                .is_some_and(|coupled| self.residual.len() == coupled)
            && self
                .kernel
                .as_ref()
                .is_some_and(|kernel| kernel.matches(dimension, shape))
    }

    /// Size stage storage and the kernel for `dimension` unknowns and `shape`.
    /// Storage is replaced only when the dimension or shape changes.
    pub(crate) fn prepare(
        &mut self,
        dimension: usize,
        shape: KernelShape<'_>,
    ) -> Result<(), SciOxideError> {
        let stages = shape.stages;
        let coupled = checked_mul(
            dimension,
            stages,
            scioxide_core::quantity!("implicit_ode_stage_dimension"),
        )?;
        if self.fits(dimension, shape) {
            return Ok(());
        }
        self.cache.reset();
        self.pending_endpoint = false;
        let kernel = StageKernel::new(dimension, shape)?;
        let radau = if shape.layout == crate::stiff::kernel::StageLayout::Radau {
            Some(crate::implicit_rk::radau_iia::RadauWorkspace::new(
                dimension,
            )?)
        } else {
            None
        };
        for descriptors in [&mut self.stages, &mut self.stage_rhs] {
            if descriptors.len() < stages {
                try_reserve(
                    descriptors,
                    stages - descriptors.len(),
                    scioxide_core::quantity!("implicit_ode_stage_descriptors"),
                )?;
                descriptors.resize_with(stages, Vector::<T>::empty);
            }
            descriptors.truncate(stages);
            descriptors
                .iter_mut()
                .try_for_each(|vector| vector.try_resize(dimension, T::ZERO))?;
        }
        for vector in [
            &mut self.current_rhs,
            &mut self.endpoint_rhs,
            &mut self.trial,
            &mut self.work,
            &mut self.weighted_rhs,
            &mut self.rhs_candidate,
        ] {
            vector.try_resize(dimension, T::ZERO)?;
        }
        for vector in [
            &mut self.coupled_state,
            &mut self.residual,
            &mut self.correction,
        ] {
            vector.try_resize(coupled, T::ZERO)?;
        }
        self.kernel = Some(kernel);
        self.radau = radau;
        self.dimension = dimension;
        Ok(())
    }

    /// Cumulative cache activity since the kernel was prepared.
    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.kernel
            .as_ref()
            .map_or_else(CacheStats::default, StageKernel::stats)
    }

    pub(crate) fn reset_epoch(&mut self) -> Result<(), SciOxideError> {
        self.cache.reset();
        self.pending_endpoint = false;
        if let Some(radau) = &mut self.radau {
            radau.reset();
        }
        self.kernel.as_mut().map_or(Ok(()), StageKernel::invalidate)
    }

    pub(crate) fn discard_step(&mut self) -> Result<(), SciOxideError> {
        self.pending_endpoint = false;
        self.kernel.as_mut().map_or(Ok(()), StageKernel::reject)
    }

    pub(crate) fn accept_step(&mut self) -> Result<(), SciOxideError> {
        if core::mem::take(&mut self.pending_endpoint) {
            core::mem::swap(&mut self.current_rhs, &mut self.endpoint_rhs);
            self.cache.advance();
        } else {
            self.cache.reset();
        }
        if let Some(radau) = &mut self.radau {
            radau.accept()?;
        }
        Ok(())
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for ImplicitOdeWorkspace<S, T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Reusable contiguous history and scratch storage for multistep methods.
/// Exceptional node geometry can require fallible extended-coefficient storage;
/// ordinary native rows do not allocate.
#[derive(Debug)]
#[must_use]
pub(crate) struct MultistepWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    pub(crate) implicit: ImplicitOdeWorkspace<S, T>,
    pub(crate) startup: ExplicitOdeWorkspace<S, T>,
    pub(crate) startup_rk4: crate::explicit_rk::ButcherTableau<S, 4>,
    pub(crate) startup_dp5: crate::explicit_rk::ButcherTableau<S, 7, 7, true>,
    pub(crate) bdf_startup: Vector<T>,
    pub(crate) bdf_coefficients: crate::multistep::bdf::Prepared<S>,
    pub(crate) adams_coefficients: crate::multistep::AdamsCoefficients<S>,
    pub(crate) current_derivative: Vector<T>,
    pub(crate) current_derivative_valid: bool,
    pub(crate) next_derivative_valid: bool,
    pub(crate) predictor: Vector<T>,
    pub(crate) corrector: Vector<T>,
    pub(crate) next: Vector<T>,
    pub(crate) rhs_sum: Vector<T>,
    pub(crate) rhs_candidate: Vector<T>,
    pub(crate) history_times: Vec<S>,
    pub(crate) history_states: Vec<T>,
    pub(crate) history_derivatives: Vec<T>,
    pub(crate) history_count: usize,
    pub(crate) history_method: Option<(&'static str, usize)>,
    pub(crate) history_step: Option<S>,
    pub(crate) history_endpoint: Option<S>,
    pub(crate) history_pending: Option<S>,
    pub(crate) accepted_endpoint: Vector<T>,
    pub(crate) history_next: usize,
    pub(crate) proposed_state: Vector<T>,
    pub(crate) errors: Vector<T>,
    pub(crate) scales: Vector<S>,
    pub(crate) error_magnitudes: Vector<S>,
    pub(crate) dense: DenseOutputStorage<S, T>,
    pub(crate) attempt_step: S,
    pub(crate) active_order: usize,
    pub(crate) next_order: usize,
    pub(crate) accepted_at_order: usize,
    pub(crate) automatic_order: bool,
    pub(crate) formula_key: u8,
    dimension: usize,
    history_capacity: usize,
    prepared: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> MultistepWorkspace<S, T> {
    pub(crate) fn reset_epoch(&mut self) -> Result<(), SciOxideError> {
        self.bdf_coefficients.invalidate();
        self.current_derivative_valid = false;
        self.next_derivative_valid = false;
        self.history_count = 0;
        self.history_next = 0;
        self.history_pending = None;
        self.history_endpoint = None;
        self.history_method = None;
        self.history_step = None;
        self.accepted_at_order = 0;
        self.active_order = 1;
        self.next_order = 1;
        self.startup.reset_epoch();
        self.implicit.reset_epoch()
    }
    pub(crate) fn accept_step(&mut self) -> Result<(), SciOxideError> {
        if let Some(time) = self.history_pending.take() {
            self.commit_predictor(time)?;
            self.current_derivative_valid = self.next_derivative_valid;
            if self.next_derivative_valid {
                self.current_derivative
                    .as_mut_slice()
                    .copy_from_slice(self.next.as_slice());
            }
            self.next_derivative_valid = false;
            self.history_endpoint = Some(time + self.attempt_step);
            self.history_step = Some(self.attempt_step);
            self.accepted_endpoint
                .as_mut_slice()
                .copy_from_slice(self.proposed_state.as_slice());
            self.accepted_at_order = if self.active_order == self.next_order {
                self.accepted_at_order.saturating_add(1)
            } else {
                0
            };
            self.active_order = self.next_order;
        }
        Ok(())
    }

    /// Create an empty multistep workspace without allocating.
    pub(crate) fn new() -> Self {
        Self {
            implicit: ImplicitOdeWorkspace::new(),
            startup: ExplicitOdeWorkspace::new(),
            startup_rk4: crate::explicit_rk::rk4::tableau(),
            startup_dp5: crate::explicit_rk::dormand_prince5::tableau(),
            bdf_startup: Vector::empty(),
            bdf_coefficients: crate::multistep::bdf::Prepared::default(),
            adams_coefficients: crate::multistep::AdamsCoefficients::default(),
            current_derivative: Vector::empty(),
            current_derivative_valid: false,
            next_derivative_valid: false,
            predictor: Vector::empty(),
            corrector: Vector::empty(),
            next: Vector::empty(),
            rhs_sum: Vector::empty(),
            rhs_candidate: Vector::empty(),
            history_times: Vec::new(),
            history_states: Vec::new(),
            history_derivatives: Vec::new(),
            history_count: 0,
            history_method: None,
            history_step: None,
            history_endpoint: None,
            history_pending: None,
            accepted_endpoint: Vector::empty(),
            history_next: 0,
            proposed_state: Vector::empty(),
            errors: Vector::empty(),
            scales: Vector::empty(),
            error_magnitudes: Vector::empty(),
            dense: DenseOutputStorage::new(),
            attempt_step: S::ZERO,
            active_order: 1,
            next_order: 1,
            accepted_at_order: 0,
            automatic_order: false,
            formula_key: 0,
            dimension: 0,
            history_capacity: 0,
            prepared: false,
        }
    }

    /// Whether `try_resize(dimension, history)` would keep this storage unchanged.
    pub(crate) const fn fits(&self, dimension: usize, history: usize) -> bool {
        self.prepared && self.dimension == dimension && self.history_capacity == history
    }

    /// Fallibly size startup scratch and the state/derivative history ring.
    ///
    /// Changing either dimension or history capacity clears accepted history.
    ///
    /// # Errors
    ///
    /// Returns [`scioxide_core::SciOxideErrorKind::SizeOverflow`] or
    /// [`scioxide_core::SciOxideErrorKind::AllocationFailure`] when storage cannot be prepared.
    pub(crate) fn try_resize(
        &mut self,
        dimension: usize,
        history: usize,
    ) -> Result<(), SciOxideError> {
        if self.fits(dimension, history) {
            return Ok(());
        }
        let history_values = checked_mul(
            dimension,
            history,
            scioxide_core::quantity!("multistep_ode_history"),
        )?;
        let two_histories = checked_mul(
            history_values,
            2,
            scioxide_core::quantity!("multistep_ode_state_history"),
        )?;
        let scratch_values = checked_mul(
            dimension,
            10,
            scioxide_core::quantity!("multistep_ode_scratch"),
        )?;
        let total = checked_add(
            two_histories,
            scratch_values,
            scioxide_core::quantity!("multistep_ode_workspace"),
        )?;
        checked_add(
            total,
            history,
            scioxide_core::quantity!("multistep_ode_workspace"),
        )?;

        // A failed resize may leave partially grown buffers. Invalidate history
        // before touching storage and publish preparation only after success.
        self.prepared = false;
        self.reset_epoch()?;

        self.accepted_endpoint.try_resize(dimension, T::ZERO)?;
        self.dense
            .try_resize(dimension, history.saturating_sub(1).max(3))?;
        self.scales.try_resize(dimension, S::ZERO)?;
        self.error_magnitudes.try_resize(dimension, S::ZERO)?;
        self.startup.try_resize(dimension, 7, 4)?;
        for vector in [
            &mut self.proposed_state,
            &mut self.errors,
            &mut self.current_derivative,
            &mut self.predictor,
            &mut self.corrector,
            &mut self.next,
            &mut self.rhs_sum,
            &mut self.rhs_candidate,
        ] {
            vector.try_resize(dimension, T::ZERO)?;
        }
        grow_vec(
            &mut self.history_times,
            history,
            S::ZERO,
            scioxide_core::quantity!("multistep_ode_time_history"),
        )?;
        grow_vec(
            &mut self.history_states,
            history_values,
            T::ZERO,
            scioxide_core::quantity!("multistep_ode_state_history"),
        )?;
        grow_vec(
            &mut self.history_derivatives,
            history_values,
            T::ZERO,
            scioxide_core::quantity!("multistep_ode_derivative_history"),
        )?;

        self.dimension = dimension;
        self.history_capacity = history;
        self.prepared = true;
        Ok(())
    }

    pub(crate) fn discard_step(&mut self) -> Result<(), SciOxideError> {
        self.history_pending = None;
        self.next_derivative_valid = false;
        self.accepted_at_order = 0;
        if self.automatic_order {
            self.active_order = self
                .next_order
                .min(self.active_order.saturating_sub(1).max(1));
        }
        self.next_order = self.active_order;
        self.implicit
            .kernel
            .as_mut()
            .map_or(Ok(()), StageKernel::reject)
    }

    pub(crate) fn history_time(&self, offset: usize) -> S {
        self.history_times[self.history_slot(offset)]
    }

    pub(crate) fn push_current(&mut self, t: S, state: &[T]) {
        self.rhs_candidate.as_mut_slice().copy_from_slice(state);
        self.history_pending = Some(t);
    }

    fn commit_predictor(&mut self, t: S) -> Result<(), SciOxideError> {
        let next_count = self.next_history_count()?;
        let slot = self.history_next;
        self.history_times[slot] = t;
        let start = slot * self.dimension;
        self.history_states[start..start + self.dimension]
            .copy_from_slice(self.rhs_candidate.as_slice());
        self.history_derivatives[start..start + self.dimension]
            .copy_from_slice(self.current_derivative.as_slice());
        let next = slot + 1;
        self.history_next = if next == self.history_capacity {
            0
        } else {
            next
        };
        self.history_count = next_count;
        Ok(())
    }

    fn next_history_count(&self) -> Result<usize, SciOxideError> {
        if self.history_count >= self.history_capacity {
            Ok(self.history_capacity)
        } else {
            self.history_count.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "multistep_ode_history_counter"
                ))
            })
        }
    }

    pub(crate) fn history_slot(&self, newest_offset: usize) -> usize {
        if newest_offset < self.history_next {
            self.history_next - 1 - newest_offset
        } else {
            self.history_capacity - (newest_offset + 1 - self.history_next)
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for MultistepWorkspace<S, T>
{
    fn default() -> Self {
        Self::new()
    }
}

fn grow_vec<S: Copy>(
    values: &mut Vec<S>,
    required: usize,
    fill: S,
    storage: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if values.len() < required {
        try_reserve(values, required - values.len(), storage)?;
    }
    values.resize(required, fill);
    Ok(())
}

#[cfg(test)]
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> MultistepWorkspace<S, T> {
    pub(crate) const fn history_capacity(&self) -> usize {
        self.history_capacity
    }
}
