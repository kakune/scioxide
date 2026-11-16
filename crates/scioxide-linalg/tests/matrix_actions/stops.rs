//! Execution stops at every provider boundary, including workspace reuse.
use super::*;
use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use scioxide_core::{
    AdjointOperator, AttemptOutcome, AttemptRecord, CancellationToken, CapabilityError, Clock,
    ClockBinding, Deadline, ExecutionStopReason, LinearOperator, MonotonicInstant, SciOxideError,
};
use scioxide_linalg::*;

use crate::support::clocks::TestClock;

struct Controlled<'a> {
    inner: &'a dyn AdjointOperator<f64>,
    calls: Cell<usize>,
    stop_at: Cell<usize>,
    provider_stop: Cell<bool>,
    failure: Cell<bool>,
    nonfinite: Cell<Option<f64>>,
    cancellation: CancellationToken,
    clock: TestClock,
}
impl<'a> Controlled<'a> {
    fn new(inner: &'a dyn AdjointOperator<f64>) -> Self {
        Self {
            inner,
            calls: Cell::new(0),
            stop_at: Cell::new(usize::MAX),
            provider_stop: Cell::new(false),
            failure: Cell::new(false),
            nonfinite: Cell::new(None),
            cancellation: CancellationToken::new(),
            clock: TestClock(AtomicU64::new(0)),
        }
    }
    fn reset(&self, stop_at: usize) {
        self.calls.set(0);
        self.stop_at.set(stop_at);
        self.cancellation.reset();
        self.clock.0.store(0, Ordering::Relaxed);
    }
    fn applied(&self, output: &mut [f64]) -> Result<(), SciOxideError> {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        if calls == self.stop_at.get() {
            if let Some(value) = self.nonfinite.get() {
                output[0] = value;
                return Ok(());
            }
            if self.failure.get() {
                return Err(SciOxideError::singular(None));
            }
            self.cancellation.cancel();
            self.clock.0.store(1, Ordering::Relaxed);
            if self.provider_stop.get() {
                return Err(SciOxideError::execution_stopped(
                    ExecutionStopReason::Cancelled,
                ));
            }
        }
        Ok(())
    }
}
impl LinearOperator<f64> for Controlled<'_> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.inner.apply_into(x, y, scratch)?;
        self.applied(y)
    }
}
impl AdjointOperator<f64> for Controlled<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.inner.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.inner.apply_adjoint_into(x, y, scratch)?;
        self.applied(y)
    }
}

fn attempt(record: Option<&AttemptRecord<()>>, termination: SpectralTermination) {
    assert_eq!(
        record.unwrap().outcome(),
        if termination == SpectralTermination::Converged {
            AttemptOutcome::Succeeded
        } else {
            AttemptOutcome::Stopped
        }
    );
}

/// Run successfully first, then pre-stop and interrupt at each provider call.
/// Both execution capabilities and stops returned directly by a provider are covered.
fn exercise(op: &Controlled<'_>, mut run: impl FnMut(ExecutionContext<'_>) -> SpectralTermination) {
    assert_eq!(
        run(ExecutionContext::sequential()),
        SpectralTermination::Converged
    );
    let calls = op.calls.get();
    assert!(calls > 0);
    op.reset(0);
    op.cancellation.cancel();
    assert_eq!(
        run(ExecutionContext::sequential().with_cancellation(&op.cancellation)),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    assert_eq!(op.calls.get(), 0);
    let binding = ClockBinding::new(&op.clock);
    let deadline = |seconds| {
        ExecutionContext::sequential()
            .with_deadline(Deadline::at(
                &binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_secs(seconds)),
            ))
            .unwrap()
    };
    op.reset(0);
    assert_eq!(
        run(deadline(0)),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::DeadlineReached)
    );
    assert_eq!(op.calls.get(), 0);
    for stop_at in 1..=calls {
        for timed in [false, true] {
            op.reset(stop_at);
            let cx = if timed {
                deadline(1)
            } else {
                ExecutionContext::sequential().with_cancellation(&op.cancellation)
            };
            assert_eq!(
                run(cx),
                SpectralTermination::ExecutionStopped(if timed {
                    ExecutionStopReason::DeadlineReached
                } else {
                    ExecutionStopReason::Cancelled
                }),
                "provider call {stop_at}"
            );
            assert_eq!(op.calls.get(), stop_at);
        }
    }
    op.provider_stop.set(true);
    op.reset(calls.div_ceil(2));
    assert_eq!(
        run(ExecutionContext::sequential()),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    op.provider_stop.set(false);
    op.reset(usize::MAX);
    assert_eq!(
        run(ExecutionContext::sequential()),
        SpectralTermination::Converged
    );
}

fn diagonal(values: &[f64]) -> Matrix<f64> {
    let mut a = Matrix::zeros(values.len(), values.len()).unwrap();
    for (i, &value) in values.iter().enumerate() {
        a.view_mut().set(i, i, value).unwrap();
    }
    a
}

#[test]
fn exponential_and_phi_stops_publish_completed_intervals() {
    let a = diagonal(&[1., 2.]);
    let finite = a.view().try_finite().unwrap();
    for phi in [0, 1] {
        let op = Controlled::new(&finite);
        let p = ExponentialAction::new(2.).with_max_phi(1);
        let mut w = p.workspace(&op, 2).unwrap();
        let input = [1.; 4];
        let mut output = [99.; 4];
        let mut saw_progress = false;
        exercise(&op, |cx| {
            output.fill(99.);
            let report = p
                .apply_into(
                    &op,
                    scioxide_linalg::PhiFunction::phi(phi, 1.),
                    BlockVectorView::new(&input, 2, 2).unwrap(),
                    BlockVectorViewMut::new(&mut output, 2, 2).unwrap(),
                    &mut w,
                    cx,
                )
                .unwrap();
            let termination = report.termination();
            assert_eq!(report.applications(), op.calls.get());
            for column in 0..2 {
                let progress = w.last_accepted().unwrap().progress()[column];
                for (row, lambda) in [1.0_f64, 2.].into_iter().enumerate() {
                    let expected = if phi == 0 {
                        (progress * lambda).exp()
                    } else {
                        (progress * lambda).exp_m1() / lambda
                    };
                    assert!((output[2 * column + row] - expected).abs() < 1e-6);
                }
                saw_progress |= matches!(termination, SpectralTermination::ExecutionStopped(_))
                    && progress > 0.;
            }
            attempt(w.last_attempt(), termination);
            termination
        });
        assert!(saw_progress);
    }
}

#[test]
fn shifted_actions_stop_at_accepted_intervals_and_preserve_output_on_failure() {
    let a = diagonal(&[-2., -1.]);
    let finite = a.view().try_finite().unwrap();
    for phi in 0..=2 {
        let op = Controlled::new(&finite);
        let options = ExponentialAction::new(0.5).with_shift(-1.5).with_max_phi(2);
        let plan = options;
        let mut workspace = plan.workspace(&op, 1).unwrap();
        let mut output = [99.; 2];
        exercise(&op, |cx| {
            let report = plan
                .apply_into(
                    &op,
                    scioxide_linalg::PhiFunction::phi(phi, 8.),
                    BlockVectorView::new(&[1.; 2], 2, 1).unwrap(),
                    BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
                    &mut workspace,
                    cx,
                )
                .unwrap();
            let fraction = workspace.last_accepted().unwrap().progress()[0];
            for (&actual, lambda) in output.iter().zip([-2.0_f64, -1.]) {
                let z = 8. * lambda;
                let expected = match phi {
                    0 => (fraction * z).exp(),
                    1 => (fraction * z).exp_m1() / z,
                    _ => ((fraction * z).exp_m1() - fraction * z) / (z * z),
                };
                assert!(
                    (actual - expected).abs() <= 1e-13,
                    "phi {phi}, fraction {fraction}: {actual} != {expected}"
                );
            }
            attempt(workspace.last_attempt(), report.termination());
            report.termination()
        });
        op.reset(1);
        op.failure.set(true);
        output.fill(99.);
        assert!(
            plan.apply_into(
                &op,
                scioxide_linalg::PhiFunction::phi(phi, 8.),
                BlockVectorView::new(&[1.; 2], 2, 1).unwrap(),
                BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .is_err()
        );
        assert_eq!(output, [99.; 2]);
        assert!(workspace.last_accepted().is_none());
    }
}

#[test]
fn function_stops_publish_completed_corrections() {
    let a = diagonal(&[1., 2., 3.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    let p = FunctionAction::default();
    let mut w = p.workspace(&op, 2).unwrap();
    let mut output = [99.; 6];
    let mut saw_progress = false;
    exercise(&op, |cx| {
        output.fill(99.);
        let report = p
            .apply_into(
                &op,
                MatrixFunction::Sqrt,
                BlockVectorView::new(&[1.; 6], 3, 2).unwrap(),
                BlockVectorViewMut::new(&mut output, 3, 2).unwrap(),
                &mut w,
                cx,
            )
            .unwrap();
        let termination = report.termination();
        assert_eq!(report.applications(), op.calls.get());
        for column in 0..2 {
            for (row, lambda) in [1.0_f64, 2., 3.].into_iter().enumerate() {
                let expected = if w.last_accepted().unwrap().completed()[column] {
                    lambda.sqrt()
                } else {
                    0.
                };
                assert!((output[column * 3 + row] - expected).abs() < 1e-8);
            }
        }
        saw_progress |=
            matches!(termination, SpectralTermination::ExecutionStopped(_)) && report.cycles() > 0;
        attempt(w.last_attempt(), termination);
        termination
    });
    assert!(saw_progress);
}

#[test]
fn spectrum_stops_publish_verified_pairs_including_shift_invert() {
    let a = diagonal(&[1., 2., 4.]);
    let inverse = diagonal(&[1., 0.5, 0.25]);
    let finite = a.view().try_finite().unwrap();
    let inverse = inverse.view().try_finite().unwrap();
    for shifted in [false, true] {
        let op = Controlled::new(&finite);
        let operator = if shifted {
            SpectrumOperator::ShiftInvert {
                a: &op,
                inverse: &inverse,
                mass: None,
                shift: 0.,
            }
        } else {
            SpectrumOperator::Direct(&op)
        };
        let p = PartialSpectrumSolver::new(2);
        let mut w = p.workspace(operator).unwrap();
        let mut output = PartialSpectrumBuffer::new(3, 2).unwrap();
        let mut saw_progress = false;
        exercise(&op, |cx| {
            let output_report = p
                .solve_into(operator, [1.; 3].as_slice(), &mut output, &mut w, cx)
                .unwrap();
            for (column, lambda) in output.eigenvalues().iter().enumerate() {
                let residual = [1., 2., 4.]
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| {
                        ((Complex::from_real(d) - *lambda)
                            * output.eigenvectors().get(i, column).unwrap())
                        .norm_sqr()
                    })
                    .sum::<f64>()
                    .sqrt();
                assert!((residual - output.residuals()[column]).abs() < 1e-10);
            }
            if op.calls.get() == 0 {
                assert!(output.eigenvalues().is_empty());
                assert!(
                    output
                        .eigenvectors()
                        .as_contiguous_slice()
                        .unwrap()
                        .iter()
                        .all(|&v| v == Complex::ZERO)
                );
            }
            saw_progress |= matches!(
                output_report.termination(),
                SpectralTermination::ExecutionStopped(_)
            ) && !output.eigenvalues().is_empty();
            assert!(output_report.applications() >= op.calls.get());
            attempt(w.last_attempt(), output_report.termination());
            output_report.termination()
        });
        assert!(saw_progress);
    }
}

#[test]
fn partial_svd_stops_publish_verified_triplets() {
    let a = diagonal(&[1., 2., 4.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    let p = PartialSvdSolver::new(2);
    let mut w = p.workspace(&op).unwrap();
    let mut output = PartialSvdBuffer::new(3, 3, 2).unwrap();
    let mut saw_progress = false;
    exercise(&op, |cx| {
        let output_report = p
            .solve_into(&op, &[1.; 3], &mut output, &mut w, cx)
            .unwrap();
        for (column, &sigma) in output.singular_values().iter().enumerate() {
            let mut residuals = [0.; 2];
            for (i, d) in [1., 2., 4.].into_iter().enumerate() {
                let u = output.left_vectors().get(i, column).unwrap();
                let v = output.right_vectors().get(i, column).unwrap();
                residuals[0] += (d * v - sigma * u).powi(2);
                residuals[1] += (d * u - sigma * v).powi(2);
            }
            for (actual, reported) in residuals.into_iter().zip(output.residuals()[column]) {
                assert!((actual.sqrt() - reported).abs() < 1e-10);
            }
        }
        if op.calls.get() == 0 {
            assert!(output.singular_values().is_empty());
            assert!(
                output
                    .left_vectors()
                    .as_contiguous_slice()
                    .unwrap()
                    .iter()
                    .chain(output.right_vectors().as_contiguous_slice().unwrap())
                    .all(|&v| v == 0.)
            );
        }
        saw_progress |= matches!(
            output_report.termination(),
            SpectralTermination::ExecutionStopped(_)
        ) && !output.singular_values().is_empty();
        assert_eq!(output_report.applications(), op.calls.get());
        attempt(w.last_attempt(), output_report.termination());
        output_report.termination()
    });
    assert!(saw_progress);
}

#[test]
fn lobpcg_stops_publish_verified_blocks() {
    let a = diagonal(&[1., 2., 3., 4.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    let p = LobpcgSolver::new(1);
    let mut w = p.workspace(&op, 0).unwrap();
    let mut output = LobpcgBuffer::new(4, 1).unwrap();
    let mut saw_progress = false;
    exercise(&op, |cx| {
        let output_report = p
            .solve_into(
                &op,
                BlockVectorView::new(&[1.; 4], 4, 1).unwrap(),
                BlockVectorView::new(&[], 4, 0).unwrap(),
                &mut output,
                &mut w,
                cx,
            )
            .unwrap();
        if output_report.iterations() == 0 {
            assert!(output.eigenvalues().is_empty());
            assert!(output.residuals().is_empty());
            assert_eq!(output.eigenvectors().cols(), 0);
        } else {
            let residual = [1., 2., 3., 4.]
                .into_iter()
                .zip(output.eigenvectors().as_contiguous_slice().unwrap())
                .map(|(d, v)| ((d - output.eigenvalues()[0]) * v).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((residual - output.residuals()[0]).abs() < 1e-10);
            saw_progress |= matches!(
                output_report.termination(),
                SpectralTermination::ExecutionStopped(_)
            );
        }
        assert_eq!(output_report.applications(), op.calls.get());
        attempt(w.last_attempt(), output_report.termination());
        output_report.termination()
    });
    assert!(saw_progress);
}

#[test]
fn randomized_stops_publish_complete_factors_and_completed_probe_diagnostics() {
    let a = diagonal(&[3., 2., 0.]);
    let finite = a.view().try_finite().unwrap();
    let sketch = Matrix::from_vec(3, 2, vec![1., 1., 1., -1., 1., 0.]).unwrap();
    let probes = Matrix::from_vec(3, 2, vec![1., 2., 3., -1., -2., 1.]).unwrap();
    for interpolative in [false, true] {
        let op = Controlled::new(&finite);
        let p = RandomizedLowRank::new(2);
        let mut w = p.workspace(&op, 2).unwrap();
        let mut svd = RandomizedSvdBuffer::new(3, 3, 2).unwrap();
        let mut id = InterpolativeBuffer::new(3, 3, 2).unwrap();
        let mut saw_progress = false;
        exercise(&op, |cx| {
            let report = if interpolative {
                p.interpolative_into(&op, sketch.view(), probes.view(), &mut id, &mut w, cx)
                    .unwrap()
            } else {
                p.svd_into(&op, sketch.view(), probes.view(), &mut svd, &mut w, cx)
                    .unwrap()
            };
            assert!(report.rank() == 0 || report.rank() == 2);
            for i in 0..3 {
                for j in 0..3 {
                    let value = (0..report.rank())
                        .map(|k| {
                            if interpolative {
                                id.columns().get(i, k).unwrap()
                                    * id.coefficients().get(k, j).unwrap()
                            } else {
                                svd.u().get(i, k).unwrap()
                                    * svd.singular_values()[k]
                                    * svd.vt().get(k, j).unwrap()
                            }
                        })
                        .sum::<f64>();
                    let expected = if report.rank() == 0 {
                        0.
                    } else {
                        a.view().get(i, j).unwrap()
                    };
                    assert!((value - expected).abs() < 1e-10);
                }
            }
            if op.calls.get() == 0 {
                assert_eq!(report.rank(), 0);
                assert_eq!(report.completed_probes(), 0);
            }
            assert!(report.probe_residual() < 1e-10);
            saw_progress |= matches!(
                report.termination(),
                SpectralTermination::ExecutionStopped(_)
            ) && report.completed_probes() > 0;
            assert_eq!(report.applications(), op.calls.get());
            attempt(w.last_attempt(), report.termination());
            report.termination()
        });
        assert!(saw_progress);
    }
}

#[test]
fn one_norm_stops_retain_the_best_completed_lower_estimate() {
    // Cover direct enumeration and the forward/adjoint iterative estimator.
    for values in [&[1., 2., 4.][..], &[1., 2., 3., 4., 8.][..]] {
        let a = diagonal(values);
        let finite = a.view().try_finite().unwrap();
        let op = Controlled::new(&finite);
        let norm = OneNormEstimator::new();
        let mut w = norm.workspace(&op).unwrap();
        let mut saw_progress = false;
        exercise(&op, |cx| {
            let report = norm.estimate_scoped(&op, &mut w, cx).unwrap();
            assert!(report.estimate() >= 0. && report.estimate() <= *values.last().unwrap());
            if op.calls.get() == 0 {
                assert_eq!(report.estimate(), 0.);
            }
            if values.len() == 3 && op.calls.get() > 0 && !op.provider_stop.get() {
                assert_eq!(report.estimate(), values[op.calls.get() - 1]);
            }
            saw_progress |= matches!(
                report.termination(),
                SpectralTermination::ExecutionStopped(_)
            ) && report.estimate() > 0.;
            assert_eq!(report.applications(), op.calls.get());
            attempt(w.last_attempt(), report.termination());
            report.termination()
        });
        assert!(saw_progress);
    }
}

#[test]
fn genuine_failures_and_invalid_inputs_remain_transactional() {
    let a = diagonal(&[1., 2., 4.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    op.failure.set(true);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    // A provider failure after one successful application, and an invalid input
    // combined with cancellation, must both preserve all public output fields.
    for invalid in [false, true] {
        let cx = if invalid {
            ExecutionContext::sequential().with_cancellation(&cancelled)
        } else {
            ExecutionContext::sequential()
        };
        let initial = if invalid { [f64::NAN; 3] } else { [1.; 3] };
        macro_rules! fails {
            ($out:expr, $workspace:expr, $call:expr) => {{
                op.reset(2);
                let before = format!("{:?}", $out);
                assert!($call.is_err());
                assert_eq!(format!("{:?}", $out), before);
                // Invalid arguments are rejected before the epoch begins.
                assert_eq!(
                    $workspace.last_attempt().unwrap().outcome(),
                    if invalid {
                        AttemptOutcome::PreflightRejected
                    } else {
                        AttemptOutcome::Failed
                    }
                );
                assert_eq!(op.calls.get(), if invalid { 0 } else { 2 });
            }};
        }
        let p = ExponentialAction::new(4.);
        let mut w = p.workspace(&op, 1).unwrap();
        let mut out = [99.; 3];
        fails!(
            out,
            w,
            p.apply_into(
                &op,
                scioxide_linalg::PhiFunction::phi(0, 1.),
                BlockVectorView::new(&initial, 3, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
                &mut w,
                cx
            )
        );

        let p = FunctionAction::default();
        let mut w = p.workspace(&op, 1).unwrap();
        fails!(
            out,
            w,
            p.apply_into(
                &op,
                MatrixFunction::Sqrt,
                BlockVectorView::new(&initial, 3, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
                &mut w,
                cx
            )
        );

        let p = PartialSpectrumSolver::new(2);
        let mut w = p.workspace(&op).unwrap();
        let mut out = PartialSpectrumBuffer::new(3, 2).unwrap();
        fails!(
            out,
            w,
            p.solve_into(&op, initial.as_slice(), &mut out, &mut w, cx)
        );
        let operator = SpectrumOperator::ShiftInvert {
            a: &finite,
            inverse: &op,
            mass: None,
            shift: 0.,
        };
        let mut w = p.workspace(operator).unwrap();
        fails!(
            out,
            w,
            p.solve_into(operator, initial.as_slice(), &mut out, &mut w, cx)
        );

        let p = PartialSvdSolver::new(2);
        let mut w = p.workspace(&op).unwrap();
        let mut out = PartialSvdBuffer::new(3, 3, 2).unwrap();
        fails!(out, w, p.solve_into(&op, &initial, &mut out, &mut w, cx));

        let p = LobpcgSolver::new(1);
        let mut w = p.workspace(&op, 0).unwrap();
        let mut out = LobpcgBuffer::new(3, 1).unwrap();
        fails!(
            out,
            w,
            p.solve_into(
                &op,
                BlockVectorView::new(&initial, 3, 1).unwrap(),
                BlockVectorView::new(&[], 3, 0).unwrap(),
                &mut out,
                &mut w,
                cx
            )
        );

        let sketch = Matrix::from_vec(
            3,
            2,
            if invalid {
                vec![f64::NAN; 6]
            } else {
                vec![1., 1., 1., -1., 1., 0.]
            },
        )
        .unwrap();
        let probes = Matrix::from_vec(3, 1, vec![1.; 3]).unwrap();
        let p = RandomizedLowRank::new(2);
        let mut w = p.workspace(&op, 2).unwrap();
        let mut out = RandomizedSvdBuffer::new(3, 3, 2).unwrap();
        fails!(
            out,
            w,
            p.svd_into(&op, sketch.view(), probes.view(), &mut out, &mut w, cx)
        );
        let mut out = InterpolativeBuffer::new(3, 3, 2).unwrap();
        fails!(
            out,
            w,
            p.interpolative_into(&op, sketch.view(), probes.view(), &mut out, &mut w, cx)
        );

        op.reset(2);
        let other = diagonal(&[1., 2.]);
        let other = other.view().try_finite().unwrap();
        let norm = OneNormEstimator::new();
        let mut w = if invalid {
            norm.workspace(&other)
        } else {
            norm.workspace(&op)
        }
        .unwrap();
        assert!(norm.estimate_scoped(&op, &mut w, cx).is_err());
        assert_eq!(
            w.last_attempt().unwrap().outcome(),
            if invalid {
                AttemptOutcome::PreflightRejected
            } else {
                AttemptOutcome::Failed
            }
        );
        assert_eq!(op.calls.get(), if invalid { 0 } else { 2 });
    }
}

#[test]
fn nonfinite_provider_output_at_every_application_is_transactional() {
    use scioxide_core::{EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let a = diagonal(&[1., 2., 4.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    macro_rules! rejects_nonfinite {
        ($out:expr,$w:expr,$cx:ident,$call:expr) => {{
            op.reset(usize::MAX);
            let $cx = ExecutionContext::sequential();
            $call.unwrap();
            let calls = op.calls.get();
            assert!(calls > 0);
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for at in 1..=calls {
                    op.reset(at);
                    op.nonfinite.set(Some(bad));
                    let before = format!("{:?}", $out);
                    let mut state = SolveState::new();
                    let mut budgets = SolveBudgets::unbounded();
                    let mut parent = state
                        .begin_solve(
                            ExecutionContext::sequential(),
                            SolveIdentity::new(
                                ProblemToken::new(1106),
                                scioxide_core::algorithm_id!("test", "nonfinite_action", 1),
                                0,
                                0,
                            ),
                            &mut budgets,
                        )
                        .unwrap();
                    let $cx = &mut parent;
                    assert_eq!(
                        $call.unwrap_err().kind(),
                        scioxide_core::SciOxideErrorKind::NonFinite
                    );
                    assert_eq!(format!("{:?}", $out), before);
                    assert_eq!($w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
                    assert_eq!(op.calls.get(), at);
                    assert_eq!(
                        parent
                            .counts()
                            .nested()
                            .provider(EvaluationRole::LinearOperator),
                        at
                    );
                    op.nonfinite.set(None);
                    op.reset(usize::MAX);
                    let $cx = ExecutionContext::sequential();
                    $call.unwrap();
                }
            }
        }};
    }
    let p = PartialSpectrumSolver::new(2);
    let mut w = p.workspace(&op).unwrap();
    let mut out = PartialSpectrumBuffer::new(3, 2).unwrap();
    rejects_nonfinite!(
        out,
        w,
        cx,
        p.solve_into(&op, [1.; 3].as_slice(), &mut out, &mut w, cx)
    );
    let p = PartialSvdSolver::new(2);
    let mut w = p.workspace(&op).unwrap();
    let mut out = PartialSvdBuffer::new(3, 3, 2).unwrap();
    rejects_nonfinite!(
        out,
        w,
        cx,
        p.solve_into(&op, &[1.; 3], &mut out, &mut w, cx)
    );
    let p = LobpcgSolver::new(1);
    let mut w = p.workspace(&op, 0).unwrap();
    let mut out = LobpcgBuffer::new(3, 1).unwrap();
    rejects_nonfinite!(
        out,
        w,
        cx,
        p.solve_into(
            &op,
            BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
            BlockVectorView::new(&[], 3, 0).unwrap(),
            &mut out,
            &mut w,
            cx
        )
    );
    let p = RandomizedLowRank::new(2);
    let mut w = p.workspace(&op, 2).unwrap();
    let sketch = Matrix::from_vec(3, 2, vec![1., 1., 1., -1., 1., 0.]).unwrap();
    let probes = Matrix::from_vec(3, 1, vec![1.; 3]).unwrap();
    let mut out = RandomizedSvdBuffer::new(3, 3, 2).unwrap();
    rejects_nonfinite!(
        out,
        w,
        cx,
        p.svd_into(&op, sketch.view(), probes.view(), &mut out, &mut w, cx)
    );
    let mut out = InterpolativeBuffer::new(3, 3, 2).unwrap();
    rejects_nonfinite!(
        out,
        w,
        cx,
        p.interpolative_into(&op, sketch.view(), probes.view(), &mut out, &mut w, cx)
    );
}

#[test]
fn stopped_shifted_inverse_does_not_start_original_pencil_verification() {
    let a = diagonal(&[1., 2., 4.]);
    let inverse = diagonal(&[1., 0.5, 0.25]);
    let a = a.view().try_finite().unwrap();
    let inverse = inverse.view().try_finite().unwrap();
    let original = Controlled::new(&a);
    let inverse = Controlled::new(&inverse);
    let operator = SpectrumOperator::ShiftInvert {
        a: &original,
        inverse: &inverse,
        mass: None,
        shift: 0.,
    };
    let p = PartialSpectrumSolver::new(2);
    let mut w = p.workspace(operator).unwrap();
    let mut out = PartialSpectrumBuffer::new(3, 2).unwrap();
    p.solve_into(
        operator,
        [1.; 3].as_slice(),
        &mut out,
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let calls = inverse.calls.get();
    assert!(calls > 1);
    original.reset(usize::MAX);
    inverse.reset(calls);
    inverse.provider_stop.set(true);
    let out_report = p
        .solve_into(
            operator,
            [1.; 3].as_slice(),
            &mut out,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        out_report.termination(),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    assert!(out.eigenvalues().is_empty());
    assert!(
        out.eigenvectors()
            .as_contiguous_slice()
            .unwrap()
            .iter()
            .all(|&v| v == Complex::ZERO)
    );
    assert_eq!(original.calls.get(), 0);
    attempt(w.last_attempt(), out_report.termination());
}

#[test]
fn one_norm_child_stops_preserve_counts_and_work_limit_estimates() {
    use scioxide_core::{EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let a = diagonal(&[1., 2., 4.]);
    let finite = a.view().try_finite().unwrap();
    let op = Controlled::new(&finite);
    let norm = OneNormEstimator::new().with_applications(Budget::ONE.saturating_add(1));
    let mut w = norm.workspace(&op).unwrap();
    for stop_at in [0, 2, usize::MAX] {
        op.reset(stop_at);
        if stop_at == 0 {
            op.cancellation.cancel();
        }
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&op.cancellation),
                SolveIdentity::new(
                    ProblemToken::new(1),
                    scioxide_core::algorithm_id!("tests", "action_parent", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let report = norm.estimate_scoped(&op, &mut w, &mut parent).unwrap();
        assert_eq!(
            report.termination(),
            if stop_at == usize::MAX {
                SpectralTermination::EvaluationLimit
            } else {
                SpectralTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
            }
        );
        assert_eq!(report.estimate(), if stop_at == 0 { 0. } else { 2. });
        assert_eq!(report.applications(), op.calls.get());
        assert_eq!(
            parent
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            report.applications()
        );
        attempt(w.last_attempt(), report.termination());
    }
}

struct Diagonal<'a>(&'a [f64]);
impl LinearOperator<f64> for Diagonal<'_> {
    fn rows(&self) -> usize {
        self.0.len()
    }
    fn cols(&self) -> usize {
        self.0.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 0);
        assert_eq!(x.len(), self.0.len());
        assert_eq!(y.len(), self.0.len());
        for ((y, x), d) in y.iter_mut().zip(x).zip(self.0) {
            *y = d * x;
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Diagonal<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}

#[test]
fn randomized_application_stops_retain_complete_factors() {
    use scioxide_core::{
        BudgetKind, EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let op = Diagonal(&[3., 2., 0.]);
    let sketch = Matrix::from_vec(3, 2, vec![1., 1., 1., -1., 1., 0.]).unwrap();
    let probes = Matrix::from_vec(3, 2, vec![1., 2., 3., -1., -2., 1.]).unwrap();
    for shared in [false, true] {
        for id in [false, true] {
            let mut previous_rank = 0;
            for budget in 1..=14 {
                let options = RandomizedLowRank::new(2).with_power_steps(1);
                let plan = if shared {
                    options
                } else {
                    options.with_applications(Budget::try_from(budget).unwrap())
                };
                let mut w = plan.workspace(&op, 2).unwrap();
                let mut svd = RandomizedSvdBuffer::new(3, 3, 2).unwrap();
                let mut interpolative = InterpolativeBuffer::new(3, 3, 2).unwrap();
                let mut state = SolveState::new();
                let mut budgets = if shared {
                    SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Evaluation, Budget::try_from(budget).unwrap())
                } else {
                    SolveBudgets::unbounded()
                };
                let mut parent = state
                    .begin_solve(
                        ExecutionContext::sequential(),
                        SolveIdentity::new(
                            ProblemToken::new(0),
                            scioxide_core::algorithm_id!("tests", "randomized_stop", 1),
                            0,
                            0,
                        ),
                        &mut budgets,
                    )
                    .unwrap();
                let report = if id {
                    plan.interpolative_into(
                        &op,
                        sketch.view(),
                        probes.view(),
                        &mut interpolative,
                        &mut w,
                        &mut parent,
                    )
                    .unwrap()
                } else {
                    plan.svd_into(
                        &op,
                        sketch.view(),
                        probes.view(),
                        &mut svd,
                        &mut w,
                        &mut parent,
                    )
                    .unwrap()
                };
                assert!(report.rank() >= previous_rank);
                previous_rank = report.rank();
                assert!(matches!(
                    report.termination(),
                    SpectralTermination::Converged | SpectralTermination::EvaluationLimit
                ));
                attempt(w.last_attempt(), report.termination());
                assert_eq!(
                    parent
                        .counts()
                        .nested()
                        .provider(EvaluationRole::LinearOperator),
                    report.applications()
                );
                assert!(report.applications() <= budget);
                for i in 0..3 {
                    for j in 0..3 {
                        let value: f64 = (0..report.rank())
                            .map(|k| {
                                if id {
                                    interpolative.columns().get(i, k).unwrap()
                                        * interpolative.coefficients().get(k, j).unwrap()
                                } else {
                                    svd.u().get(i, k).unwrap()
                                        * svd.singular_values()[k]
                                        * svd.vt().get(k, j).unwrap()
                                }
                            })
                            .sum();
                        let expected = if report.rank() == 0 || i != j {
                            0.
                        } else {
                            op.0[i]
                        };
                        assert!((value - expected).abs() < 1e-12);
                    }
                }
            }
            assert_eq!(previous_rank, 2);
        }
    }
}

#[test]
fn spectral_budget_sweeps_never_shrink_verified_results() {
    use scioxide_core::{
        BudgetKind, EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let n = 200;
    let diagonal: Vec<_> = (0..n)
        .map(|i| {
            if i < 6 {
                30. - i as f64
            } else {
                1. + i as f64 / n as f64
            }
        })
        .collect();
    let inverse: Vec<_> = diagonal.iter().map(|d| 1. / d).collect();
    let ones = vec![1.; n];
    let op = Diagonal(&diagonal);
    let inverted = Diagonal(&inverse);
    for family in 0..4 {
        let mut previous_count = 0;
        let options = PartialSpectrumSolver::new(6)
            .with_basis(Budget::try_from(20).unwrap())
            .with_structure(if family == 1 {
                OperatorStructure::Hermitian
            } else {
                OperatorStructure::General
            });
        let eig = options;
        let operator = if family == 3 {
            SpectrumOperator::ShiftInvert {
                a: &inverted,
                inverse: &op,
                mass: None,
                shift: 0.,
            }
        } else {
            SpectrumOperator::Direct(&op)
        };
        let mut ew = eig.workspace(operator).unwrap();
        let svd = PartialSvdSolver::new(6).with_basis(Budget::try_from(20).unwrap());
        let mut sw = svd.workspace(&op).unwrap();
        let mut pairs = PartialSpectrumBuffer::new(n, 6).unwrap();
        let mut triples = PartialSvdBuffer::new(n, n, 6).unwrap();
        for budget in 1..=180 {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(budget).unwrap());
            let mut parent = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    SolveIdentity::new(
                        ProblemToken::new(0),
                        scioxide_core::algorithm_id!("tests", "spectral_sweep", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            let (count, applications, termination) = if family == 2 {
                let triples_report = svd
                    .solve_into(&op, &ones, &mut triples, &mut sw, &mut parent)
                    .unwrap();
                attempt(sw.last_attempt(), triples_report.termination());
                (
                    triples.singular_values().len(),
                    triples_report.applications(),
                    triples_report.termination(),
                )
            } else {
                let pairs_report = eig
                    .solve_into(operator, ones.as_slice(), &mut pairs, &mut ew, &mut parent)
                    .unwrap();
                attempt(ew.last_attempt(), pairs_report.termination());
                (
                    pairs.eigenvalues().len(),
                    pairs_report.applications(),
                    pairs_report.termination(),
                )
            };
            assert!(
                count >= previous_count,
                "family={family} budget={budget}: {previous_count} -> {count}"
            );
            previous_count = count;
            assert_eq!(
                parent
                    .counts()
                    .nested()
                    .provider(EvaluationRole::LinearOperator),
                applications
            );
            assert!(applications <= budget);
            assert!(matches!(
                termination,
                SpectralTermination::Converged | SpectralTermination::EvaluationLimit
            ));
            if family != 2 {
                let ds = if family == 3 { &inverse } else { &diagonal };
                for (column, &lambda) in pairs.eigenvalues().iter().enumerate() {
                    let actual = ds
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| {
                            ((Complex::from_real(d) - lambda)
                                * pairs.eigenvectors().get(i, column).unwrap())
                            .norm_sqr()
                        })
                        .sum::<f64>()
                        .sqrt();
                    assert!((actual - pairs.residuals()[column]).abs() < 1e-12);
                }
            }
        }
        assert_eq!(previous_count, 6);
    }
}

#[test]
fn restarted_shift_invert_accepts_original_pencil_and_charges_once() {
    use scioxide_core::{
        BudgetKind, EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let n = 40;
    let initial = vec![1.; n];
    for with_mass in [false, true] {
        let eigenvalues: Vec<_> = (0..n).map(|i| 1. + i as f64 / 3.).collect();
        let mass: Vec<_> = (0..n)
            .map(|i| {
                if with_mass {
                    1. + i as f64 / n as f64
                } else {
                    1.
                }
            })
            .collect();
        let a: Vec<_> = eigenvalues.iter().zip(&mass).map(|(l, m)| l * m).collect();
        let inv: Vec<_> = a.iter().map(|a| 1. / a).collect();
        let (a, inv, mass) = (Diagonal(&a), Diagonal(&inv), Diagonal(&mass));
        let operator = SpectrumOperator::ShiftInvert {
            a: &a,
            inverse: &inv,
            mass: with_mass.then_some(&mass as &dyn LinearOperator<f64>),
            shift: 0.,
        };
        let options = PartialSpectrumSolver::new(2).with_basis(Budget::try_from(8).unwrap());
        let tolerance = options.tolerance();
        let plan = options;
        let mut w = plan.workspace(operator).unwrap();
        let mut out = PartialSpectrumBuffer::new(n, 2).unwrap();
        for budget in [5, 9, 20, 40, 1000] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(budget).unwrap());
            let mut parent = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    SolveIdentity::new(
                        ProblemToken::new(0),
                        scioxide_core::algorithm_id!("tests", "shift_budget", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            let out_report = plan
                .solve_into(operator, initial.as_slice(), &mut out, &mut w, &mut parent)
                .unwrap();
            assert_eq!(
                out_report.applications(),
                parent
                    .counts()
                    .nested()
                    .provider(EvaluationRole::LinearOperator)
            );
            attempt(w.last_attempt(), out_report.termination());
            if budget == 1000 {
                assert_eq!(out_report.termination(), SpectralTermination::Converged);
                assert!(out_report.cycles() > 1);
                for (column, &lambda) in out.eigenvalues().iter().enumerate() {
                    let anorm = (0..n)
                        .map(|i| {
                            (a.0[i] * out.eigenvectors().get(i, column).unwrap().norm()).powi(2)
                        })
                        .sum::<f64>()
                        .sqrt();
                    assert!(out.residuals()[column] <= tolerance.threshold(anorm).unwrap());
                    assert!((lambda.re - (column as f64 / 3. + 1.)).abs() < 1e-12);
                }
            } else if budget <= 9 {
                assert_eq!(
                    out_report.termination(),
                    SpectralTermination::EvaluationLimit
                );
            }
        }
    }
}

#[test]
fn partial_svd_nested_factor_uses_action_accounting_and_deadline() {
    struct FactorClock {
        applications: AtomicU64,
        checks: AtomicU64,
        stop: bool,
    }
    impl Clock for FactorClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            let seconds = u64::from(
                self.stop
                    && self.applications.load(Ordering::Relaxed) >= 16
                    && self.checks.fetch_add(1, Ordering::Relaxed) >= 40,
            );
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(seconds),
            ))
        }
    }
    struct TimedDiagonal<'a> {
        values: Diagonal<'a>,
        clock: &'a FactorClock,
    }
    impl LinearOperator<f64> for TimedDiagonal<'_> {
        fn rows(&self) -> usize {
            self.values.rows()
        }
        fn cols(&self) -> usize {
            self.values.cols()
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], s: &mut [f64]) -> Result<(), SciOxideError> {
            self.values.apply_into(x, y, s)?;
            self.clock.applications.fetch_add(1, Ordering::Relaxed);
            self.clock.checks.store(0, Ordering::Relaxed);
            Ok(())
        }
    }
    impl AdjointOperator<f64> for TimedDiagonal<'_> {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.apply_into(x, y, s)
        }
    }
    let values: Vec<_> = (0..32).map(|i| 1. + i as f64).collect();
    for stop in [false, true] {
        let clock = FactorClock {
            applications: AtomicU64::new(0),
            checks: AtomicU64::new(0),
            stop,
        };
        let binding = ClockBinding::new(&clock);
        let op = TimedDiagonal {
            values: Diagonal(&values),
            clock: &clock,
        };
        let plan = PartialSvdSolver::new(2)
            .with_basis(Budget::try_from(8).unwrap())
            .with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new()
                    .with_time_limit(Duration::from_secs(1)),
            );
        let mut w = plan.workspace(&op).unwrap();
        let mut out = PartialSvdBuffer::new(32, 32, 2).unwrap();
        let out_report = plan
            .solve_into(
                &op,
                &[1.; 32],
                &mut out,
                &mut w,
                ExecutionContext::sequential().with_clock(&binding).unwrap(),
            )
            .unwrap();
        assert!(w.last_attempt().unwrap().counts().nested().iterations() > 0);
        if stop {
            assert_eq!(
                out_report.termination(),
                SpectralTermination::ExecutionStopped(ExecutionStopReason::DeadlineReached)
            );
            assert_eq!(
                out_report.applications(),
                16,
                "deadline must interrupt the first small SVD before verification probes"
            );
        } else {
            assert_eq!(out_report.termination(), SpectralTermination::Converged);
        }
        attempt(w.last_attempt(), out_report.termination());
    }
}
