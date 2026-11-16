//! Provider and CPU backend admission, resource accounting and transactional execution.
#![expect(clippy::unwrap_used)]
#[path = "support/executors.rs"]
mod executors;
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/owned.rs"]
mod owned;
use owned::ExecuteOwned;
#[path = "support/provider.rs"]
mod provider;
use executors::BrokenExecutor;

use provider::{DirectProvider, Mode};
use scioxide_core::Executor;
use scioxide_core::backend::KernelBackend;
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar, SequentialExecutor,
    StridedLayout,
};
use scioxide_fft::*;
use std::sync::atomic::Ordering;

fn execution() -> ExecutionContext<'static> {
    ExecutionContext::new(&SequentialExecutor)
}
fn context<S: RealScalar>(p: &DirectProvider) -> FftPlanningContext<'_, S>
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    FftPlanningContext::with_provider(execution(), p, FftProviderPolicy::Required)
}
fn signal<S: RealScalar>(n: usize) -> Vec<Complex<S>>
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    (0..n)
        .map(|i| {
            Complex::new(
                S::try_from_f64_lossy((i as f64 * 0.73).sin()).unwrap(),
                S::try_from_f64_lossy((i as f64 * 0.31).cos()).unwrap(),
            )
        })
        .collect()
}
// Provider parity on one transform, whose work is the compared length.
fn close<S: RealScalar>(a: &[Complex<S>], b: &[Complex<S>]) {
    numerics::assert_fft_close(a, b, a.len());
}
fn spec(
    n: usize,
    direction: FftDirection,
    normalization: Normalization,
    layout: FftLayout,
) -> FftPlanSpec {
    FftPlanSpec::new(n, direction, normalization, layout)
        .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar))
}
fn parity<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let provider = DirectProvider::new(Mode::Direct);
    for n in [1, 2, 3, 4, 7, 16, 17, 23, 46, 60] {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for normalization in [
                Normalization::Unscaled,
                Normalization::Forward,
                Normalization::Inverse,
                Normalization::Orthonormal,
            ] {
                let spec = spec(n, direction, normalization, FftLayout::Complex);
                let cpu =
                    FftPlan::build(spec, scioxide_fft::FftPlanningContext::default()).unwrap();
                let external = FftPlan::build(spec, context(&provider)).unwrap();
                assert_eq!(
                    external.resolved_backends().next().unwrap(),
                    if n == 1 {
                        FftBackend::Cpu(KernelBackend::Scalar)
                    } else {
                        FftBackend::Provider
                    }
                );
                let mut actual = signal::<S>(n);
                let mut expected = actual.clone();
                let mut work = external.workspace(()).unwrap();
                external
                    .execute_in_place(
                        &mut actual,
                        &mut work,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap();
                cpu.execute_in_place(
                    &mut expected,
                    &mut cpu.workspace(()).unwrap(),
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
                close(&actual, &expected);
            }
        }
    }
    assert!(provider.calls.load(Ordering::Relaxed) > 0);
}
#[test]
fn scalar_provider_parity_both_precisions() {
    parity::<f32>();
    parity::<f64>();
}

#[test]
fn failures_are_transactional_and_fallback_is_only_for_unsupported_preparation() {
    let spec = spec(
        17,
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    );
    for mode in [
        Mode::Decline,
        Mode::PreparationFailure,
        Mode::ExecutionFailure,
        Mode::NonFinite,
        Mode::ExcessScratch,
    ] {
        let p = DirectProvider::new(mode);
        let required = FftPlan::<f64>::build(spec, context(&p));
        let preferred = FftPlan::<f64>::build(
            spec,
            FftPlanningContext::with_provider(execution(), &p, FftProviderPolicy::Preferred),
        );
        match mode {
            Mode::Decline => {
                assert!(required.is_err());
                assert_eq!(
                    preferred.unwrap().resolved_backends().next().unwrap(),
                    FftBackend::Cpu(KernelBackend::Scalar)
                );
            }
            Mode::PreparationFailure | Mode::ExcessScratch => {
                assert!(required.is_err());
                assert!(preferred.is_err());
            }
            _ => {
                let plan = required.unwrap();
                let mut input = signal::<f64>(17);
                let original = input.clone();
                let error = plan
                    .execute_in_place(
                        &mut input,
                        &mut plan.workspace(()).unwrap(),
                        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                    )
                    .unwrap_err();
                if matches!(mode, Mode::NonFinite) {
                    assert!(matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NonFinite {
                            origin: scioxide_core::NonFiniteOrigin::Callback,
                            ..
                        }
                    ));
                } else {
                    assert!(matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::ExecutionFailure { .. }
                    ));
                }
                assert_eq!(input, original);
                assert!(preferred.is_ok());
            }
        }
    }
    let p = DirectProvider::new(Mode::Direct);
    let plan = FftPlan::<f64>::build(spec, context(&p)).unwrap();
    let mut data = signal::<f64>(17);
    let original = data.clone();
    assert!(
        plan.execute_in_place(
            &mut data,
            &mut vec![
                Complex::new(0.0, 0.0);
                plan.workspace_requirements(()).unwrap().complex_len() - 1
            ],
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(data, original);
    assert_eq!(p.calls.load(Ordering::Relaxed), 0);
    assert!(
        FftPlan::<f64>::build(
            spec,
            context(&p).with_provider_limits(FftProviderLimits {
                preparation_bytes: 0,
                workspace_bytes: 0
            })
        )
        .is_err()
    );
}

#[test]
fn provider_plans_do_not_use_cpu_cache() {
    let p = DirectProvider::new(Mode::Direct);
    let mut cache = FftPlanCache::<f64>::new(FftCachePolicy::new(4, 1 << 20).unwrap()).unwrap();
    for _ in 0..2 {
        let _ = cache
            .plan(
                spec(
                    7,
                    FftDirection::Forward,
                    Normalization::Inverse,
                    FftLayout::Complex,
                ),
                context(&p),
            )
            .unwrap();
    }
    assert_eq!(p.preparations.load(Ordering::Relaxed), 2);
}

fn real_families<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let p = DirectProvider::new(Mode::Direct);
    for n in [3, 4, 17, 24, 45] {
        let input: Vec<S> = signal::<S>(n).iter().map(|z| z.re).collect();
        for norm in [
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Orthonormal,
            Normalization::Unscaled,
        ] {
            let f = FftPlan::build(
                spec(n, FftDirection::Forward, norm, FftLayout::Real),
                context(&p),
            )
            .unwrap();
            let inv = FftPlan::build(
                spec(n, FftDirection::Inverse, norm, FftLayout::Real),
                context(&p),
            )
            .unwrap();
            let mut bins = HermitianSpectrum::zeros(n).unwrap();
            let mut scratch = f.workspace(()).unwrap();
            f.execute_into(
                FftInput::from(&input),
                scioxide_fft::FftOutput::Hermitian {
                    logical_len: bins.logical_len(),
                    bins: scioxide_linalg::VectorViewMut::from_slice(bins.bins_mut()),
                },
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
            let cpu = rfft(&input, norm).unwrap();
            close(bins.bins(), cpu.bins());
            let mut out = vec![S::ZERO; n];
            inv.execute_into(
                scioxide_fft::FftInput::Hermitian {
                    logical_len: bins.logical_len(),
                    bins: scioxide_linalg::VectorView::from_slice(bins.bins()),
                },
                FftOutput::from(&mut out),
                &mut inv.workspace(()).unwrap(),
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
            let expected = irfft(&cpu, norm).unwrap();
            close(
                &out.iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
                &expected
                    .iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
            );
        }
        macro_rules! check {
            ($plan:ident, $spec:path, $ty:ident) => {
                for ty in [$ty::I, $ty::II, $ty::III, $ty::IV] {
                    for direction in [FftDirection::Forward, FftDirection::Inverse] {
                        let cpu = $plan::new(<$spec>::new(
                            n,
                            ty.into(),
                            direction,
                            Normalization::Orthonormal,
                        ))
                        .unwrap();
                        let external = $plan::build(
                            <$spec>::new(n, ty.into(), direction, Normalization::Orthonormal)
                                .with_execution_policy(ExecutionPolicy::Sequential),
                            context(&p),
                        )
                        .unwrap();
                        assert_eq!(
                            external.resolved_backends().next().unwrap(),
                            FftBackend::Provider
                        );
                        let mut out = vec![S::ZERO; n];
                        let mut expected = out.clone();
                        let mut scratch = external.workspace(()).unwrap();
                        external
                            .execute_into(
                                &input,
                                &mut out,
                                &mut scratch,
                                ExecutionContext::sequential(),
                            )
                            .unwrap();
                        cpu.execute_into(
                            &input,
                            &mut expected,
                            &mut cpu.workspace(()).unwrap(),
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                        close(
                            &out.iter()
                                .map(|&x| Complex::new(x, S::ZERO))
                                .collect::<Vec<_>>(),
                            &expected
                                .iter()
                                .map(|&x| Complex::new(x, S::ZERO))
                                .collect::<Vec<_>>(),
                        );
                    }
                }
            };
        }
        check!(
            RealTransformPlan,
            scioxide_fft::RealTransformPlanSpec,
            DctType
        );
        check!(
            RealTransformPlan,
            scioxide_fft::RealTransformPlanSpec,
            DstType
        );
    }
}
#[test]
fn real_and_symmetry_reductions_use_provider() {
    real_families::<f32>();
    real_families::<f64>();
}

#[test]
fn strided_batches_and_zoom_use_provider() {
    let p = DirectProvider::new(Mode::Direct);
    let spec = FftNdPlanSpec::new(
        &[3, 4],
        &[0, 1],
        FftDirection::Forward,
        Normalization::Orthonormal,
        FftLayout::Complex,
    )
    .unwrap();
    let plan = FftNdPlan::build(spec.clone(), context(&p)).unwrap();
    assert!(plan.resolved_backends().all(|b| b == FftBackend::Provider));
    let cpu = FftNdPlan::new(spec).unwrap();
    let data = signal::<f64>(24);
    let layout = StridedLayout::new(&[3, 4], &[8, 2]).unwrap();
    let view = NdView::new(&data, layout).unwrap();
    let out_layout = StridedLayout::new(&[3, 4], &[4, 1]).unwrap();
    let mut actual = vec![Complex::new(0.0, 0.0); 12];
    let mut expected = actual.clone();
    plan.execute_into(
        FftNdInput::Complex(NdView::new(&data, layout).unwrap()),
        FftNdOutput::Complex(NdViewMut::new(&mut actual, out_layout).unwrap()),
        &mut plan.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    cpu.execute_into(
        FftNdInput::Complex(view),
        FftNdOutput::Complex(NdViewMut::new(&mut expected, out_layout).unwrap()),
        &mut cpu.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    close(&actual, &expected);
    let zoom = CztPlan::build(
        CztPlanSpec::new(
            12,
            7,
            CztContour::zoom(0.0, 0.5, 7, 1.0, false).unwrap(),
            Normalization::Inverse,
        ),
        context(&p),
    )
    .unwrap();
    let cpu = CztPlan::new(CztPlanSpec::new(
        12,
        7,
        CztContour::zoom(0.0, 0.5, 7, 1.0, false).unwrap(),
        Normalization::Inverse,
    ))
    .unwrap();
    assert_eq!(zoom.resolved_backends().next(), Some(FftBackend::Provider));
    let mut actual = vec![Complex::new(0.0, 0.0); 7];
    let mut expected = actual.clone();
    zoom.execute_into(
        &data[..12],
        &mut actual,
        &mut zoom.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    cpu.execute_into(
        &data[..12],
        &mut expected,
        &mut cpu.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    close(&actual, &expected);
}

#[test]
fn short_nd_axes_keep_provider_execution_and_failure_checks() {
    let shape = [2, 257];
    let layout = StridedLayout::new(&shape, &[257, 1]).unwrap();
    let input = signal::<f64>(514);
    let spec = FftNdPlanSpec::new(
        &shape,
        &[0, 1],
        FftDirection::Forward,
        Normalization::Orthonormal,
        FftLayout::Complex,
    )
    .unwrap();
    let cpu = FftNdPlan::new(spec.clone()).unwrap();
    let mut expected = input.clone();
    cpu.execute_in_place(
        NdViewMut::new(&mut expected, layout).unwrap(),
        &mut cpu.workspace(()).unwrap(),
        execution(),
    )
    .unwrap();
    for mode in [Mode::Direct, Mode::ExecutionFailure, Mode::NonFinite] {
        let provider = DirectProvider::new(mode);
        let plan = FftNdPlan::build(spec.clone(), context(&provider)).unwrap();
        let sentinel = Complex::new(123.0, -321.0);
        let mut output = vec![sentinel; input.len()];
        let result = plan.execute_into(
            FftNdInput::Complex(NdView::new(&input, layout).unwrap()),
            FftNdOutput::Complex(NdViewMut::new(&mut output, layout).unwrap()),
            &mut plan.workspace(()).unwrap(),
            execution(),
        );
        assert!(provider.calls.load(Ordering::Relaxed) > 0);
        if matches!(mode, Mode::Direct) {
            result.unwrap();
            assert_eq!(provider.calls.load(Ordering::Relaxed), 259);
            close(&output, &expected);
        } else {
            assert!(result.is_err());
            assert!(output.iter().all(|&v| v == sentinel));
        }
    }
}

fn nonuniform<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let provider = DirectProvider::new(Mode::Direct);
    let coordinates: Vec<S> = [0.1, 0.7, -1.5]
        .into_iter()
        .map(|x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let frequencies: Vec<S> = [-1.0, 0.25, 1.5, 2.0]
        .into_iter()
        .map(|x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let pshape = [3, 1];
    let fshape = [4, 1];
    let strides = [1, 1];
    let points =
        || NdView::new(&coordinates, StridedLayout::new(&pshape, &strides).unwrap()).unwrap();
    let freq =
        || NdView::new(&frequencies, StridedLayout::new(&fshape, &strides).unwrap()).unwrap();
    let eps = if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
        1e-3
    } else {
        1e-7
    };
    let options = NufftPlanSpec::new(
        scioxide_fft::FftDirection::Forward,
        scioxide_fft::Normalization::Unscaled,
        S::try_from_f64_lossy(eps).unwrap(),
    )
    .with_limits(NufftLimits {
        preparation_bytes: 16 << 20,
        workspace_bytes: 16 << 20,
        grid_points: 1 << 20,
        stencil_visits: 1 << 20,
        kernel_support: 128,
    });
    for kind in 1..=3 {
        let build = |ctx| match kind {
            1 => NufftPlan::build(
                scioxide_fft::NufftTransform::Type1 {
                    points: points(),
                    modes: &[4],
                },
                options,
                ctx,
            ),
            2 => NufftPlan::build(
                scioxide_fft::NufftTransform::Type2 {
                    points: points(),
                    modes: &[4],
                },
                options,
                ctx,
            ),
            _ => NufftPlan::build(
                scioxide_fft::NufftTransform::Type3 {
                    points: points(),
                    frequencies: freq(),
                },
                options,
                ctx,
            ),
        };
        let external = build(context(&provider)).unwrap();
        let cpu = build(execution().into()).unwrap();
        assert!(
            external
                .resolved_backends()
                .all(|b| b == FftBackend::Provider)
        );
        let input = signal::<S>(external.input_len());
        let mut actual = vec![Complex::new(S::ZERO, S::ZERO); external.output_len()];
        let mut scratch = external.workspace(()).unwrap();
        external
            .execute_into(
                &input,
                &mut actual,
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
        close(&actual, &cpu.execute_owned(&input).unwrap());
    }
    assert!(provider.calls.load(Ordering::Relaxed) >= 3);
}
#[test]
fn nonuniform_provider_both_precisions() {
    nonuniform::<f32>();
    nonuniform::<f64>();
}

#[cfg(feature = "std")]
#[test]
fn prepared_cpu_capabilities_execute_with_sequential_context_and_match_provider() {
    let p = DirectProvider::new(Mode::Direct);
    let native = execution().with_processor(&scioxide_core::StdProcessor);
    let request = spec(
        60,
        FftDirection::Forward,
        Normalization::Orthonormal,
        FftLayout::Complex,
    );
    let direct = FftPlan::build(request, context(&p)).unwrap();
    let mut expected = signal::<f64>(60);
    direct
        .execute_in_place(
            &mut expected,
            &mut direct.workspace(()).unwrap(),
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    for backend in numerics::fft_backends(native.processor()) {
        let cpu = FftPlan::build(
            request.with_backend_strategy(FftBackendStrategy::Required(backend)),
            native,
        )
        .unwrap();
        let mut actual = signal::<f64>(60);
        let mut scratch = cpu.workspace(()).unwrap();
        cpu.execute_in_place(&mut actual, &mut scratch, ExecutionContext::sequential())
            .unwrap();
        close(&actual, &expected);
    }
}

#[test]
fn lying_provider_cpu_dispatch_and_external_fallback() {
    fn check<S: RealScalar>()
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        struct Liar;
        impl scioxide_core::ProcessorCapabilities for Liar {
            fn supports(&self, _: scioxide_core::ProcessorFeature) -> bool {
                true
            }
        }
        let execution = execution().with_processor(&Liar);
        for n in [16, 17, 23, 64, 125] {
            let request = spec(
                n,
                FftDirection::Forward,
                Normalization::Unscaled,
                FftLayout::Complex,
            )
            .with_backend_strategy(FftBackendStrategy::Automatic);
            let direct = DirectProvider::new(Mode::Direct);
            let reference = FftPlan::<S>::build(request, context(&direct)).unwrap();
            let mut expected = signal::<S>(n);
            reference
                .execute_in_place(
                    &mut expected,
                    &mut reference.workspace(()).unwrap(),
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
            let plan = FftPlan::<S>::build(request, execution).unwrap();
            let mut actual = signal::<S>(n);
            let mut scratch = plan.workspace(()).unwrap();
            plan.execute_in_place(&mut actual, &mut scratch, execution)
                .unwrap();
            close(&actual, &expected);
            let decline = DirectProvider::new(Mode::Decline);
            let unsupported = |result: Result<_, scioxide_core::SciOxideError>| {
                matches!(
                    result
                        .map(|_: scioxide_fft::FftPlan<S>| ())
                        .unwrap_err()
                        .view(),
                    scioxide_core::SciOxideErrorView::InvalidInput {
                        reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                        ..
                    }
                )
            };
            for backend in [
                KernelBackend::Scalar,
                KernelBackend::Avx2Fma,
                KernelBackend::Avx512,
                KernelBackend::Neon,
            ] {
                let forced = request.with_backend_strategy(FftBackendStrategy::Required(backend));
                let result = FftPlan::<S>::build(forced, execution);
                if backend == KernelBackend::Avx512 {
                    // FFT has no AVX-512 leaves: the request is rejected even
                    // when the processor claims support and even when a
                    // provider would serve the length or decline it.
                    assert!(unsupported(result));
                    for provider in [&direct, &decline] {
                        assert!(unsupported(FftPlan::<S>::build(
                            forced,
                            FftPlanningContext::with_provider(
                                execution,
                                provider,
                                FftProviderPolicy::Preferred,
                            ),
                        )));
                    }
                } else if backend.is_available(&Liar) {
                    let plan = result.unwrap();
                    actual = signal(n);
                    plan.execute_in_place(&mut actual, &mut plan.workspace(()).unwrap(), execution)
                        .unwrap();
                    close(&actual, &expected);
                } else {
                    assert!(unsupported(result));
                    // An external implementation requires no CPU proof; a declined
                    // preparation must verify the requested CPU fallback.
                    let external = FftPlan::<S>::build(
                        forced,
                        FftPlanningContext::with_provider(
                            execution,
                            &direct,
                            FftProviderPolicy::Preferred,
                        ),
                    )
                    .unwrap();
                    assert_eq!(
                        external.resolved_backends().next().unwrap(),
                        FftBackend::Provider
                    );
                    assert!(
                        FftPlan::<S>::build(
                            forced,
                            FftPlanningContext::with_provider(
                                execution,
                                &decline,
                                FftProviderPolicy::Preferred
                            )
                        )
                        .is_err()
                    );
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn executor_protocol_failures_preserve_output() {
    for executor in BrokenExecutor::ALL {
        let ctx = ExecutionContext::new(&executor);
        let request = spec(
            65536,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        )
        .with_execution_policy(ExecutionPolicy::Parallel(executor.worker_budget()));
        let plan = FftPlan::build(request, ctx).unwrap();
        let mut data = signal::<f64>(65536);
        let original = data.clone();
        assert!(
            plan.execute_in_place(&mut data, &mut plan.workspace(()).unwrap(), ctx)
                .is_err()
        );
        assert_eq!(data, original);
    }
    let p = DirectProvider::new(Mode::Direct);
    let plan = FftPlan::build(
        spec(
            7,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        context(&p),
    )
    .unwrap();
    let cancel = scioxide_core::CancellationToken::new();
    cancel.cancel();
    let mut data = signal::<f64>(7);
    let original = data.clone();
    assert!(
        plan.execute_in_place(
            &mut data,
            &mut plan.workspace(()).unwrap(),
            execution().with_cancellation(&cancel)
        )
        .is_err()
    );
    assert_eq!(data, original);
    assert_eq!(p.calls.load(Ordering::Relaxed), 0);
}

#[path = "support/cpu_provider.rs"]
mod cpu_provider;
#[test]
fn injected_provider_executor_parity_allocations_and_failure() {
    use std::num::NonZeroUsize;
    let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap());
    let adapter = cpu_provider::CpuProvider::<f64>::new(65536, policy).unwrap();
    let request = spec(
        65536,
        FftDirection::Forward,
        Normalization::Orthonormal,
        FftLayout::Complex,
    )
    .with_execution_policy(policy);
    let external = FftPlan::build(
        request,
        FftPlanningContext::with_provider(execution(), &adapter, FftProviderPolicy::Required),
    )
    .unwrap();
    let mut work = external.workspace(()).unwrap();
    let original = signal::<f64>(65536);
    for executor in BrokenExecutor::ALL {
        let mut data = original.clone();
        assert!(
            external
                .execute_in_place(&mut data, &mut work, ExecutionContext::new(&executor))
                .is_err()
        );
        assert_eq!(data, original);
    }
    let mut data = original.clone();
    external
        .execute_in_place(&mut data, &mut work, execution())
        .unwrap();
    #[cfg(feature = "std")]
    let mut parallel = original.clone();
    let mut partitioned = original;
    external
        .execute_in_place(
            &mut partitioned,
            &mut work,
            ExecutionContext::new(&InlinePartitions),
        )
        .unwrap();
    close(&partitioned, &data);
    #[cfg(feature = "std")]
    {
        let pool = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
        external
            .execute_in_place(&mut parallel, &mut work, ExecutionContext::new(&pool))
            .unwrap();
        close(&parallel, &data);
    }
}

struct InlinePartitions;
impl scioxide_core::Executor for InlinePartitions {
    fn worker_budget(&self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        scioxide_core::Executor::execute(&SequentialExecutor, context, plan, operation)
    }
}

fn packed_batches<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let provider = DirectProvider::new(Mode::Direct);
    let shape = [2, 3, 5];
    let strides = [30, 10, 2];
    let packed = [2, 3, 3];
    let packed_strides = [9, 3, 1];
    let dense = [15, 5, 1];
    let source: Vec<S> = signal::<S>(60).iter().map(|z| z.re).collect();
    let input_layout = StridedLayout::new(&shape, &strides).unwrap();
    let output_layout = StridedLayout::new(&packed, &packed_strides).unwrap();
    let real_layout = StridedLayout::new(&shape, &dense).unwrap();
    for family in [FftLayout::Real, FftLayout::Hermitian] {
        for norm in [
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Orthonormal,
            Normalization::Unscaled,
        ] {
            let direction = if family == FftLayout::Real {
                FftDirection::Forward
            } else {
                FftDirection::Inverse
            };
            let request = FftNdPlanSpec::new(&shape, &[1, 2], direction, norm, family).unwrap();
            let external = FftNdPlan::build(request.clone(), context(&provider)).unwrap();
            let cpu = FftNdPlan::build(request, execution()).unwrap();
            let mut actual = vec![Complex::new(S::ZERO, S::ZERO); 18];
            let mut expected = actual.clone();
            let mut scratch = external.workspace(()).unwrap();
            (|| {
                external.execute_into(
                    FftNdInput::Real(NdView::new(&source, input_layout)?),
                    FftNdOutput::Complex(NdViewMut::new(&mut actual, output_layout)?),
                    &mut scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
            })()
            .unwrap();
            cpu.execute_into(
                FftNdInput::Real(NdView::new(&source, input_layout).unwrap()),
                FftNdOutput::Complex(NdViewMut::new(&mut expected, output_layout).unwrap()),
                &mut cpu.workspace(()).unwrap(),
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
            close(&actual, &expected);
            let inverse_direction = if direction == FftDirection::Forward {
                FftDirection::Inverse
            } else {
                FftDirection::Forward
            };
            let inverse = FftNdPlan::build(
                FftNdPlanSpec::new(&shape, &[1, 2], inverse_direction, norm, family).unwrap(),
                context(&provider),
            )
            .unwrap();
            let mut restored = vec![S::ZERO; 30];
            let mut work = inverse.workspace(()).unwrap();
            inverse
                .execute_into(
                    FftNdInput::Complex(NdView::new(&actual, output_layout).unwrap()),
                    FftNdOutput::Real(NdViewMut::new(&mut restored, real_layout).unwrap()),
                    &mut work,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
            let gain = if norm == Normalization::Unscaled {
                S::try_from_f64_lossy(15.0).unwrap()
            } else {
                S::ONE
            };
            let expected: Vec<_> = NdView::new(&source, input_layout)
                .unwrap()
                .iter()
                .map(|&x| Complex::new(x * gain, S::ZERO))
                .collect();
            close(
                &restored
                    .iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
                &expected,
            );
            // The packed DC self-conjugate point must remain real, even for providers.
            actual[0].im = S::try_from_f64_lossy(100.0).unwrap();
            let original = restored.clone();
            assert!(
                inverse
                    .execute_into(
                        FftNdInput::Complex(NdView::new(&actual, output_layout).unwrap()),
                        FftNdOutput::Real(NdViewMut::new(&mut restored, real_layout).unwrap()),
                        &mut work,
                        scioxide_fft::FftPlanningContext::<f64>::default().execution()
                    )
                    .is_err()
            );
            assert_eq!(restored, original);
        }
    }
}
#[test]
fn real_hermitian_strided_batches_both_precisions() {
    packed_batches::<f32>();
    packed_batches::<f64>();
}

#[test]
fn provider_preparation_peak_must_cover_retained_bytes_and_fit_its_cap() {
    for (mode, kind) in [
        (
            Mode::UnderreportedPeak,
            scioxide_core::SciOxideErrorKind::InvalidInput,
        ),
        (
            Mode::ExcessPreparation,
            scioxide_core::SciOxideErrorKind::ResourceLimit,
        ),
    ] {
        let provider = DirectProvider::new(mode);
        let result = NufftPlan::build(
            NufftTransform::Type1 {
                points: NdView::new(&[0.0_f64], StridedLayout::new(&[1, 1], &[1, 1]).unwrap())
                    .unwrap(),
                modes: &[3],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6),
            context(&provider).with_provider_limits(FftProviderLimits {
                preparation_bytes: 1 << 20,
                workspace_bytes: 1 << 20,
            }),
        );
        let error = result.unwrap_err();
        assert_eq!(error.kind(), kind);
        if matches!(mode, Mode::UnderreportedPeak) {
            assert!(
                matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("context"))
            );
        }
    }
}

struct ElevenProvider(DirectProvider);
impl<S: RealScalar> FftProvider<S> for ElevenProvider {
    fn prepare(
        &self,
        length: usize,
        policy: ExecutionPolicy,
        limits: FftProviderLimits,
        cx: ExecutionContext<'_>,
    ) -> Result<Option<std::sync::Arc<dyn FftProviderPlan<S>>>, scioxide_core::SciOxideError> {
        if length == 11 {
            self.0.prepare(length, policy, limits, cx)
        } else {
            Ok(None)
        }
    }
}

fn real_residue_provider<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_core::{NonFiniteOrigin, NumericalIssue, SciOxideError, SciOxideErrorView};
    let scalar = |value| S::try_from_f64_lossy(value).unwrap();
    for mode in [Mode::Direct, Mode::NonFinite, Mode::LargeFinite] {
        let provider = ElevenProvider(DirectProvider::new(mode));
        let cx =
            FftPlanningContext::with_provider(execution(), &provider, FftProviderPolicy::Preferred);
        let request = spec(
            99,
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Real,
        );
        let plan = FftPlan::build(request, cx).unwrap();
        let backends = [
            FftBackend::Cpu(KernelBackend::Scalar),
            FftBackend::Provider,
            FftBackend::Provider,
        ];
        assert!(plan.resolved_backends().eq(backends));
        {
            assert!(plan.resolved_backends().eq(backends));
        };
        let mut input: Vec<_> = signal::<S>(99).into_iter().map(|value| value.re).collect();
        if matches!(mode, Mode::LargeFinite) {
            input.fill(scalar(4294967296.0));
        }
        let sentinel = Complex::new(scalar(17.0), scalar(-23.0));
        let mut output = vec![sentinel; 50];
        let result = plan.execute_into(
            &input[..],
            FftOutput::Hermitian {
                logical_len: 99,
                bins: scioxide_linalg::VectorViewMut::from_slice(&mut output),
            },
            &mut plan.workspace(()).unwrap(),
            execution(),
        );
        let check_error = |error: SciOxideError| match mode {
            Mode::NonFinite => assert!(
                matches!(
                    error.view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Callback,
                        ..
                    }
                ),
                "{error:?}"
            ),
            Mode::LargeFinite => assert!(
                matches!(
                    error.view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ),
                "{error:?}"
            ),
            _ => unreachable!(),
        };
        if matches!(mode, Mode::Direct) {
            result.unwrap();
            close(
                &output,
                rfft(&input, Normalization::Unscaled).unwrap().bins(),
            );
        } else {
            check_error(result.unwrap_err());
            assert!(output.iter().all(|&value| value == sentinel));
        }
        assert!(provider.0.calls.load(Ordering::Relaxed) > 0);

        let dct = RealTransformPlan::build(
            RealTransformPlanSpec::new(
                99,
                DctType::II.into(),
                FftDirection::Forward,
                Normalization::Unscaled,
            )
            .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar)),
            cx,
        )
        .unwrap();
        assert!(dct.resolved_backends().eq(backends));
        let mut real_output = vec![scalar(17.0); 99];
        let result = dct.execute_into(
            &input,
            &mut real_output,
            &mut dct.workspace(()).unwrap(),
            execution(),
        );
        if matches!(mode, Mode::Direct) {
            result.unwrap();
            let expected = scioxide_fft::dct(&input, DctType::II, Normalization::Unscaled).unwrap();
            close(
                &real_output
                    .iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
                &expected
                    .iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
            );
        } else {
            check_error(result.unwrap_err());
            assert!(real_output.iter().all(|&value| value == scalar(17.0)));
        }

        let nd = FftNdPlan::build(
            FftNdPlanSpec::new(
                &[1, 99],
                &[0, 1],
                FftDirection::Forward,
                Normalization::Unscaled,
                FftLayout::Real,
            )
            .unwrap()
            .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar)),
            cx,
        )
        .unwrap();
        assert!(nd.resolved_backends().any(|b| b == FftBackend::Provider));
        output.fill(sentinel);
        let result = nd.execute_into(
            FftNdInput::Real(
                NdView::new(&input, StridedLayout::new(&[1, 99], &[99, 1]).unwrap()).unwrap(),
            ),
            FftNdOutput::Complex(
                NdViewMut::new(&mut output, StridedLayout::new(&[1, 50], &[50, 1]).unwrap())
                    .unwrap(),
            ),
            &mut nd.workspace(()).unwrap(),
            execution(),
        );
        if matches!(mode, Mode::Direct) {
            result.unwrap();
            close(
                &output,
                rfft(&input, Normalization::Unscaled).unwrap().bins(),
            );
        } else {
            check_error(result.unwrap_err());
            assert!(output.iter().all(|&value| value == sentinel));
        }
    }
}

#[test]
fn composite_real_plans_report_and_check_every_provider_child() {
    real_residue_provider::<f32>();
    real_residue_provider::<f64>();
}

struct IgnoresLimits(provider::DirectProvider);

impl scioxide_fft::FftProvider<f64> for IgnoresLimits {
    fn prepare(
        &self,
        n: usize,
        policy: scioxide_core::ExecutionPolicy,
        _: scioxide_fft::FftProviderLimits,
        cx: ExecutionContext<'_>,
    ) -> Result<
        Option<std::sync::Arc<dyn scioxide_fft::FftProviderPlan<f64>>>,
        scioxide_core::SciOxideError,
    > {
        self.0
            .prepare(n, policy, scioxide_fft::FftProviderLimits::default(), cx)
    }
}

#[test]
fn provider_limits_are_structured_and_retained_capacity_is_charged() {
    use scioxide_fft::{FftPlanningContext, FftProviderLimits, FftProviderPolicy};
    let provider = IgnoresLimits(provider::DirectProvider::new(provider::Mode::Direct));
    let spec = FftPlanSpec::new(
        4,
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    );
    for limits in [
        FftProviderLimits {
            preparation_bytes: 1,
            workspace_bytes: usize::MAX,
        },
        FftProviderLimits {
            preparation_bytes: usize::MAX,
            workspace_bytes: 1,
        },
    ] {
        let error = FftPlan::build(
            spec,
            FftPlanningContext::with_provider(
                ExecutionContext::sequential(),
                &provider,
                FftProviderPolicy::Required,
            )
            .with_provider_limits(limits),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        let scioxide_core::SciOxideErrorView::ResourceLimit {
            resource: _view_resource,
            required: _view_required,
            limit: _view_limit,
        } = error.view()
        else {
            unreachable!("unexpected error kind")
        };

        assert_eq!(_view_limit, 1);
        assert!(_view_required > _view_limit);
    }
    let plan = FftPlan::build(
        spec,
        FftPlanningContext::with_provider(
            ExecutionContext::sequential(),
            &provider,
            FftProviderPolicy::Required,
        )
        .with_provider_limits(FftProviderLimits {
            preparation_bytes: usize::MAX,
            workspace_bytes: 4 * size_of::<Complex<f64>>(),
        }),
    )
    .unwrap();
    let requirements = plan.workspace_requirements(()).unwrap();
    let mut values = Vec::with_capacity(requirements.complex_len() + 1);
    values.resize(requirements.complex_len(), Complex::new(0.0, 0.0));
    let mut workspace = FftWorkspace::from_vec(values);
    assert_eq!(
        workspace.check(&requirements).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        workspace.try_reserve(&requirements).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let mut oversized_scratch = vec![Complex::new(0.0, 0.0); requirements.complex_len() + 1];
    let input = [Complex::new(1.0, 0.0); 4];
    let mut output = [Complex::new(0.0, 0.0); 4];
    plan.execute_into(
        &input,
        &mut output,
        &mut oversized_scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut in_place = input;
    plan.execute_in_place(
        &mut in_place,
        &mut oversized_scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn provider_plan_outlives_construction_capabilities() {
    let plan = {
        let provider = provider::DirectProvider::new(provider::Mode::Direct);
        FftPlan::build(
            FftPlanSpec::new(
                4,
                FftDirection::Forward,
                Normalization::Inverse,
                FftLayout::Complex,
            ),
            scioxide_fft::FftPlanningContext::with_provider(
                ExecutionContext::sequential(),
                &provider,
                scioxide_fft::FftProviderPolicy::Required,
            ),
        )
        .unwrap()
    };
    let input = [Complex::new(1.0_f64, 0.0); 4];
    let mut output = [Complex::new(0.0, 0.0); 4];
    plan.execute_into(
        &input,
        &mut output,
        &mut plan.workspace(()).unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((output[0].re - 4.0).abs() < 1e-12);
    assert!(output[1..].iter().all(|value| value.norm() < 1e-12));
}

#[test]
fn inverse_real_provider_residual_failures_leave_output_unchanged() {
    use scioxide_fft::{FftLayout, FftNdInput, FftNdOutput, FftNdPlan, FftNdPlanSpec};
    use scioxide_linalg::VectorView;
    let provider = provider::DirectProvider::new(provider::Mode::ImaginaryResidual);
    let context = scioxide_fft::FftPlanningContext::with_provider(
        ExecutionContext::sequential(),
        &provider,
        scioxide_fft::FftProviderPolicy::Required,
    );
    let plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            7,
            FftDirection::Inverse,
            Normalization::Inverse,
            FftLayout::Real,
        ),
        context,
    )
    .unwrap();
    let mut output = [41.0; 7];
    let bins = [Complex::new(1.0, 0.0); 4];
    let error = plan
        .execute_into(
            FftInput::Hermitian {
                logical_len: 7,
                bins: VectorView::from_slice(&bins),
            },
            &mut output,
            &mut plan.workspace(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(output, [41.0; 7]);
    let shape = [3, 5];
    let packed_shape = [3, 3];
    let bins = [Complex::new(1.0, 0.0); 9];
    for (family, direction) in [
        (FftLayout::Real, FftDirection::Inverse),
        (FftLayout::Hermitian, FftDirection::Forward),
    ] {
        let plan = FftNdPlan::<f64>::build(
            FftNdPlanSpec::new(&shape, &[0, 1], direction, Normalization::Inverse, family).unwrap(),
            context,
        )
        .unwrap();
        let mut output = [41.0; 15];
        let error = plan
            .execute_into(
                FftNdInput::Complex(
                    scioxide_core::NdView::new(
                        &bins,
                        scioxide_core::StridedLayout::new(&packed_shape, &[3, 1]).unwrap(),
                    )
                    .unwrap(),
                ),
                FftNdOutput::Real(
                    scioxide_core::NdViewMut::new(
                        &mut output,
                        scioxide_core::StridedLayout::new(&shape, &[5, 1]).unwrap(),
                    )
                    .unwrap(),
                ),
                &mut plan.workspace(()).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ));
        assert_eq!(output, [41.0; 15]);
    }
}
