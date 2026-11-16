//! Saved copies of a continuation state.
use super::*;

/// Stores a copy of a [`ContinuationState`] for [`Continuation::resume`].
///
/// The copy includes the accepted points, the tangent and the work counted so
/// far, so that limits from [`ContinuationOptions::with_budgets`] apply across
/// the restart. Create one with [`ContinuationState::checkpoint`].
pub struct ContinuationCheckpoint<R: RealScalar> {
    pub(crate) state: ContinuationState<R>,
    pub(crate) metadata: SolveCheckpointMetadata,
}
impl<R: RealScalar> ContinuationCheckpoint<R> {
    /// Returns the work counts and solve identity stored in the checkpoint.
    ///
    /// The execution context is not stored; [`Continuation::resume`] takes a new one.
    pub fn metadata(&self) -> SolveCheckpointMetadata {
        self.metadata
    }
}

impl<R: RealScalar> ContinuationState<R> {
    pub(crate) fn state_token(&self) -> u128 {
        crate::context::options_token(&(
            &self.point,
            &self.residual,
            self.report,
            &self.tangent,
            self.history_token,
            self.next_target,
            self.distance,
            self.next_step,
            self.status,
        ))
    }
    /// Copies this state into a checkpoint.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if no point has been accepted yet.
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn checkpoint(&self) -> Result<ContinuationCheckpoint<R>, SciOxideError> {
        let metadata = self.metadata.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "no accepted state to checkpoint",
            )
        })?;
        let path = match &self.path {
            ContinuationPath::Parameters(v) => {
                ContinuationPath::Parameters(try_copy_slice(v, scioxide_core::quantity!("path"))?)
            }
            ContinuationPath::PseudoArclength(s) => ContinuationPath::PseudoArclength(*s),
        };
        let mut points = scioxide_core::backend::try_with_capacity(
            self.points.capacity(),
            scioxide_core::quantity!("history_points"),
        )?;
        points.extend_from_slice(&self.points);
        let mut history = scioxide_core::backend::try_with_capacity(
            self.options.max_points.get(),
            scioxide_core::quantity!("history"),
        )?;
        history.extend_from_slice(&self.history);
        let state = Self {
            path,
            options: self.options.clone(),
            point: try_copy_slice(&self.point, scioxide_core::quantity!("point"))?,
            residual: try_copy_slice(
                &self.residual,
                scioxide_core::quantity!("accepted_residual"),
            )?,
            report: self.report,
            tangent: try_copy_slice(&self.tangent, scioxide_core::quantity!("tangent"))?,
            points,
            history,
            history_token: self.history_token,
            next_target: self.next_target,
            distance: self.distance,
            next_step: self.next_step,
            status: self.status,
            metadata: Some(metadata),
            provider_shape: self.provider_shape,
        };
        Ok(ContinuationCheckpoint { state, metadata })
    }
}

impl<R: RealScalar> ContinuationCheckpoint<R> {
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        let state = &self.state;
        if state.point.is_empty()
            || state.residual.len() != state.point.len() - 1
            || state.report.is_none()
            || state.points.len()
                != state
                    .point
                    .len()
                    .checked_mul(state.history.len())
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("history_points"))
                    })?
            || state.state_token() != self.metadata.state_token()
            || state.identity() != self.metadata.identity()
        {
            return Err(invalid(
                scioxide_core::argument!("checkpoint"),
                InvalidInputReason::Inconsistent,
                "checkpoint configuration or payload integrity mismatch",
            ));
        }
        let history = state
            .points
            .chunks_exact(state.point.len())
            .zip(&state.history)
            .fold(0, |token, (point, observation)| {
                crate::context::options_token(&(token, point, Some(observation)))
            });
        if history != state.history_token {
            return Err(invalid(
                scioxide_core::argument!("checkpoint"),
                InvalidInputReason::Inconsistent,
                "checkpoint history integrity mismatch",
            ));
        }
        finite(&state.point, NonFiniteOrigin::Input)?;
        finite(&state.residual, NonFiniteOrigin::Input)?;
        finite(&state.tangent, NonFiniteOrigin::Input)
    }
}
