//! Allocation, shape, cancellation, borrowed-axis, workspace and executor contracts.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{
    CancellationToken, Capacity, Complex, ExecutionContext, ExecutionPlan, ExecutionPolicy,
    Executor, PartitionFn, SciOxideError, SciOxideErrorKind, SequentialExecutor,
};
use scioxide_signal::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

fn plan() -> SpectralEstimatePlan<f64> {
    let o = SpectralEstimateOptions::new()
        .with_fft_len(Some(7))
        .with_sample_rate(2.)
        .with_detrend(None)
        .with_overlap(2);
    SpectralEstimatePlan::build(&[0.2, 0.5, 1., 0.5, 0.2], o, ExecutionContext::sequential())
        .unwrap()
}

#[test]
fn failures_never_publish() {
    let p = plan();
    let mut work = p.workspace(SignalLayout::Contiguous(8)).unwrap();
    let mut out = vec![123.; p.bins()];
    let mut input = vec![1.; 8];
    input[2] = f64::INFINITY;
    assert_eq!(
        error_kind(p.power_into(&input, &mut out, &mut work, ExecutionContext::sequential())),
        SciOxideErrorKind::NonFinite
    );
    assert!(out.iter().all(|&x| x == 123.));
    let mut small = SignalWorkspace::new();
    assert_eq!(
        p.power_into(
            &[1.; 8],
            &mut out,
            &mut small,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        error_kind(p.coherence_into(
            &[0.; 8],
            &[0.; 8],
            &mut out,
            &mut work,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::DomainError
    );
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert_eq!(
        error_kind(p.power_into(
            &[1.; 8],
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&cancel)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert!(out.iter().all(|&x| x == 123.));
    let times = [0., 0.3, 1.2, 2.9];
    let f = [1., 2.];
    let p = LombScargle::new(&times, &f, None, Default::default()).unwrap();
    let mut work = p.workspace(1).unwrap();
    let mut out = [123.; 2];
    assert_eq!(
        error_kind(p.power_into(
            &[0.; 4],
            &mut out,
            &mut work,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        error_kind(p.power_into(
            &[1., 2., 3., 4.],
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&cancel)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, [123.; 2]);
    let same = [1.; 4];
    let p = LombScargle::new(&same, &f, None, Default::default()).unwrap();
    p.power_into(
        &[1., 2., 3., 4.],
        &mut out,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out, [0.; 2]);
}
#[test]
fn workspace_requirements_reserve_check_and_limit() {
    let p = plan();
    let requirements = p
        .workspace_requirements(SignalLayout::Contiguous(12))
        .unwrap();
    assert_eq!(requirements.values_len(), 3 * p.segment_len());
    assert_eq!(
        error_kind(p.workspace_requirements(SignalLayout::Contiguous(4))),
        SciOxideErrorKind::InvalidInput
    );
    // An axis layout adds two lanes and the staged bins of every lane.
    let axis = p
        .workspace_requirements(SignalLayout::Axis {
            shape: &[3, 12],
            axis: 1,
        })
        .unwrap();
    assert_eq!(axis.values_len(), 2 * 12 + requirements.values_len());
    assert_eq!(
        axis.complex_len(),
        3 * p.bins() + requirements.complex_len()
    );
    let mut ws = SignalWorkspace::new();
    assert_eq!(ws.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    ws.try_reserve(&requirements).unwrap();
    assert_eq!(ws.check(&requirements).unwrap(), Capacity::Sufficient);
    let x: Vec<_> = (0..12).map(|i| (i as f64 * 0.6).sin()).collect();
    let mut out = vec![0.; p.bins()];
    p.power_into(&x, &mut out, &mut ws, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(out, p.power(&x).unwrap());
    let limited = requirements.with_max_bytes(requirements.bytes::<f64>().unwrap() - 1);
    assert_eq!(
        ws.check(&limited).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        SignalWorkspace::<f64>::new()
            .try_reserve(&limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let times = [0., 0.4, 1.1, 2.3, 3.0];
    let f = [0.9, 1.7];
    let lomb = LombScargle::new(&times, &f, None, Default::default()).unwrap();
    // `lanes·M` staged values, one lane, one result and the `4N` kernel.
    for lanes in [0, 1, 7] {
        assert_eq!(
            lomb.workspace_requirements(lanes).unwrap().values_len(),
            lanes * f.len() + times.len() + f.len() + 4 * times.len()
        );
    }
    assert_eq!(
        error_kind(lomb.workspace_requirements(usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
}
#[test]
fn invalid_domains_and_sizes() {
    let o = SpectralEstimateOptions::new()
        .with_fft_len(Some(4))
        .with_sample_rate(1.);
    assert_eq!(
        error_kind(SpectralEstimatePlan::<f64>::new(&[], o)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SpectralEstimatePlan::<f64>::new(&[0.; 4], o)),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        error_kind(SpectralEstimatePlan::<f64>::new(
            &[1.; 4],
            o.with_overlap(4)
        )),
        SciOxideErrorKind::InvalidInput
    );
    let o = o.with_packing(StftPacking::OneSided);
    assert_eq!(
        error_kind(SpectralEstimatePlan::<Complex<f64>>::new(&[1.; 4], o)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SpectralEstimatePlan::<f64>::new(
            &[1.; 4],
            SpectralEstimateOptions::new()
                .with_fft_len(Some(4))
                .with_sample_rate(0.)
                .with_packing(StftPacking::OneSided)
        )),
        SciOxideErrorKind::InvalidInput
    );
    let o = o.with_scaling(SpectralScaling::Spectrum);
    assert_eq!(
        error_kind(SpectralEstimatePlan::<f64>::new(&[1., -1.], o)),
        SciOxideErrorKind::DomainError
    );
    let mean = plan();
    assert_eq!(
        mean.workspace_requirements(SignalLayout::Contiguous(usize::MAX))
            .unwrap(),
        mean.workspace_requirements(SignalLayout::Contiguous(5))
            .unwrap()
    );
    let median_options = mean.options().with_average(SpectralAverage::Median);
    let median =
        SpectralEstimatePlan::<f64>::new(&[0.2, 0.5, 1., 0.5, 0.2], median_options).unwrap();
    assert!(matches!(
        median
            .workspace_requirements(SignalLayout::Contiguous(usize::MAX))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
    assert_eq!(
        error_kind(plan().workspace_requirements(SignalLayout::Axis {
            shape: &[5],
            axis: 1
        })),
        SciOxideErrorKind::InvalidInput
    );
    let t = [0., 1., 2.];
    let f = [1.];
    assert_eq!(
        error_kind(LombScargle::new(
            &t,
            &[0.],
            None,
            LombScargleOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(LombScargle::new(
            &t,
            &f,
            Some(&[0.; 3]),
            LombScargleOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(LombScargle::new(
            &t,
            &f,
            Some(&[1., -1., 1.]),
            LombScargleOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(LombScargle::new(
            &t[..2],
            &f,
            None,
            LombScargleOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert!(
        LombScargle::new(
            &t[..2],
            &f,
            None,
            LombScargleOptions::default().with_mean(LombScargleMean::Zero)
        )
        .is_ok()
    );
}
struct Counting(AtomicUsize);
impl Executor for Counting {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        SequentialExecutor.execute(context, plan, operation)
    }
}
#[test]
fn executor_reaches_fft_and_preserves_order() {
    let executor = Counting(AtomicUsize::new(0));
    let context = ExecutionContext::new(&executor);
    let o = SpectralEstimateOptions::new()
        .with_fft_len(Some(131072))
        .with_sample_rate(1.)
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()))
        .with_detrend(None);
    let p = SpectralEstimatePlan::<f64>::build(&[1.; 8], o, context).unwrap();
    let input = [1., 2., 0., 3., -1., 0.1, 0.2, 0.3];
    let mut out = vec![0.; p.bins()];
    let mut work = p.workspace(SignalLayout::Contiguous(8)).unwrap();
    p.power_into(&input, &mut out, &mut work, context).unwrap();
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    let reference = SpectralEstimatePlan::<f64>::new(
        &[1.; 8],
        o.with_execution_policy(ExecutionPolicy::Sequential),
    )
    .unwrap()
    .power(&input)
    .unwrap();
    let error = out
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(error < 2e-12, "executor difference {error}");
}

#[test]
fn mean_scales_segments_before_rounding_overflowing_power() {
    let options = SpectralEstimateOptions::new()
        .with_fft_len(Some(1))
        .with_sample_rate(1.0)
        .with_detrend(None);
    let plan = SpectralEstimatePlan::<f64>::new(&[1.0], options).unwrap();
    // The first segment's power overflows, but the three-segment mean is finite.
    let result = plan.power(&[1.5e154, 0.0, 0.0]).unwrap();
    assert!((result[0] / 7.5e307 - 1.0).abs() < 8.0 * f64::EPSILON);
    let mut out = [123.0];
    let mut scratch = plan.workspace(SignalLayout::Contiguous(3)).unwrap();
    let error = plan
        .power_into(
            &[f64::MAX, 0.0, 0.0],
            &mut out,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(out, [123.0]);
}
