//! Provider-raised stops retain only verified corrections and their diagnostics.
use super::*;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::time::Duration;
use scioxide_core::{
    CancellationToken, CapabilityError, Clock, ClockBinding, Deadline, ExecutionStopReason,
    MonotonicInstant, NonFiniteOrigin, OperatorContext,
};

struct Signal {
    cancellation: CancellationToken,
    time: AtomicU64,
    reason: ExecutionStopReason,
    live: bool,
}
impl Clock for Signal {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            Duration::from_secs(self.time.load(Ordering::Relaxed)),
        ))
    }
}
impl Signal {
    fn new(reason: ExecutionStopReason, live: bool) -> Self {
        Self {
            cancellation: CancellationToken::new(),
            time: AtomicU64::new(0),
            reason,
            live,
        }
    }
    fn raise(&self) {
        if self.live {
            if self.reason == ExecutionStopReason::Cancelled {
                self.cancellation.cancel();
            } else {
                self.time.store(1, Ordering::Relaxed);
            }
        }
    }
}

struct Provider<'a, S> {
    signal: Option<&'a Signal>,
    calls: AtomicUsize,
    stop_at: usize,
    singular_at: usize,
    factor: S,
}
impl<'a, S: NumericScalar> Provider<'a, S> {
    fn new(factor: S, signal: Option<&'a Signal>, stop_at: usize) -> Self {
        Self {
            signal,
            calls: AtomicUsize::new(0),
            stop_at,
            singular_at: usize::MAX,
            factor,
        }
    }
    fn write(&self, input: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        let calls = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        for (out, value) in output.iter_mut().zip(input) {
            *out = *value * self.factor;
        }
        if calls == self.singular_at {
            return Err(SciOxideError::singular(None));
        }
        if calls == self.stop_at {
            // A stopped provider may leave incomplete/nonfinite candidate data.
            output[0] = S::from_real(S::Real::NAN);
            if let Some(signal) = self.signal {
                signal.raise();
                return Ok(true);
            }
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Callback,
                "injected provider failure",
            ));
        }
        Ok(false)
    }
}
impl<S: NumericScalar> LinearOperator<S> for Provider<'_, S> {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, input: &[S], output: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        if self.write(input, output)? {
            return Err(SciOxideError::execution_stopped(
                self.signal.unwrap().reason,
            ));
        }
        Ok(())
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        _: &mut [S],
        context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        if self.write(input, output)? {
            let signal = self.signal.unwrap();
            if signal.live {
                // Exercise exactly the provider-side execution check used by LU.
                context.execution().check()?;
            }
            return Err(SciOxideError::execution_stopped(signal.reason));
        }
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Low,
    Fallback,
    Residual(usize),
}

#[test]
fn later_column_stop_publishes_earlier_results_and_preserves_unstarted_columns() {
    for reason in [
        ExecutionStopReason::Cancelled,
        ExecutionStopReason::DeadlineReached,
    ] {
        let signal = Signal::new(reason, true);
        let binding = ClockBinding::new(&signal);
        let execution = ExecutionContext::sequential()
            .with_cancellation(&signal.cancellation)
            .with_deadline(Deadline::at(
                &binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
            ))
            .unwrap();
        let inverse = Provider::new(1_f32, Some(&signal), 2);
        let a = Matrix::<f64>::identity(2).unwrap();
        let problem =
            RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None)
                .unwrap();
        let plan = IterativeRefinement::new(options()).unwrap();
        let mut workspace = plan
            .workspace(&RefinementWorkspaceRequest::new(&problem).with_right_hand_sides(3))
            .unwrap();
        let rhs = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let mut x = Matrix::from_vec(2, 3, vec![0.25; 6]).unwrap();
        let reports = plan
            .refine_into(
                &problem,
                rhs.view(),
                x.view_mut(),
                &mut workspace,
                execution,
            )
            .unwrap();
        assert!(reports[0].is_success());
        assert_eq!(
            reports[1].termination(),
            RefinementTermination::ExecutionStopped(reason)
        );
        assert_eq!(
            reports[2].termination(),
            RefinementTermination::ExecutionStopped(reason)
        );
        assert!(reports[1].final_error().is_some());
        assert!(reports[2].final_error().is_none());
        assert_eq!(x.as_slice(), [1., 0.25, 0.25, 4., 0.25, 0.25]);
        assert_eq!(inverse.calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    }
}

#[test]
fn non_stop_provider_errors_roll_back_output_and_fail_the_attempt() {
    for stage in [
        Stage::Low,
        Stage::Fallback,
        Stage::Residual(1),
        Stage::Residual(3),
    ] {
        let mut low = Provider::new(
            0.5_f32,
            None,
            if matches!(stage, Stage::Low) {
                2
            } else {
                usize::MAX
            },
        );
        if matches!(stage, Stage::Fallback) {
            low.singular_at = 2;
        }
        let high = Provider::new(0.5_f64, None, 2);
        let residual = Provider::new(
            1_f64,
            None,
            if let Stage::Residual(at) = stage {
                at
            } else {
                usize::MAX
            },
        );
        let problem = RefinementProblem::new(
            RefinementSystem::Operator {
                operator: &residual,
                norm_bound: 1.,
            },
            &low,
            None,
            Some(&high),
        )
        .unwrap();
        let plan = IterativeRefinement::new(
            options()
                .with_criterion(RefinementCriterion::Normwise)
                .with_fallback(RefinementFallback::OnFailure),
        )
        .unwrap();
        let mut workspace = plan
            .workspace(&RefinementWorkspaceRequest::new(&problem))
            .unwrap();
        let rhs = Vector::from_vec(vec![1., 2.]);
        let mut x = Vector::from_vec(vec![0.25, 0.5]);
        let error = plan
            .refine_into(
                &problem,
                rhs.view(),
                x.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite, "{stage:?}");
        assert_eq!(x.as_slice(), [0.25, 0.5]);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
}
