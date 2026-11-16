//! Deterministic planning, workspace alignment and caller-owned cache contracts.
#![expect(clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_core::backend::KernelBackend;
use scioxide_core::{Complex, ExecutionPolicy, SciOxideError};
use scioxide_fft::{
    FftBackendStrategy, FftCachePolicy, FftDirection, FftLayout, FftPlan, FftPlanCache,
    FftPlanSpec, Normalization,
};

fn signal(length: usize) -> Vec<Complex<f64>> {
    (0..length)
        .map(|index| {
            let position = index as f64 + 1.0;
            Complex::new(
                (0.37 * position).sin() - 0.2 * (0.11 * position).cos(),
                0.5 * (0.23 * position).cos() + 0.1 * (0.19 * position).sin(),
            )
        })
        .collect()
}

fn scalar_plan<S: scioxide_core::RealScalar>(length: usize) -> FftPlan<S> {
    FftPlan::<S>::build(
        FftPlanSpec::new(
            length,
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        )
        .with_execution_policy(ExecutionPolicy::Sequential)
        .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar)),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap()
}

#[test]
fn construction_is_deterministic_and_scratch_obeys_alignment() {
    let first = scalar_plan::<f64>(210);
    let second = scalar_plan::<f64>(210);
    assert_eq!(first.spec(), second.spec());
    assert_eq!(
        first.resolved_backends().next().unwrap(),
        second.resolved_backends().next().unwrap()
    );
    assert_eq!(first.diagnostics(), second.diagnostics());
    assert_eq!(first.plan_bytes(), second.plan_bytes());
    assert_eq!(
        first.workspace_requirements(()).unwrap(),
        second.workspace_requirements(()).unwrap()
    );

    let mut first_output = signal(210);
    let mut second_output = first_output.clone();
    let mut scratch = first.workspace(()).unwrap();
    assert_eq!(
        scratch.as_ptr() as usize % first.workspace_requirements(()).unwrap().alignment(),
        0
    );
    first
        .execute_in_place(
            &mut first_output,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    second
        .execute_in_place(
            &mut second_output,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    assert_eq!(first_output, second_output);
}

#[test]
fn failed_planning_preserves_caller_owned_cache() -> Result<(), SciOxideError> {
    let policy = FftCachePolicy::new(2, 8 * 1024 * 1024)?;
    let mut cache = FftPlanCache::<f64>::new(policy)?;
    let valid = FftPlanSpec::new(
        64,
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    );
    let invalid = FftPlanSpec::new(
        0,
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    );
    {
        let _ = cache.plan(valid, scioxide_fft::FftPlanningContext::default())?;
    }
    let before = cache.stats();
    {
        let error = cache
            .plan(invalid, scioxide_fft::FftPlanningContext::default())
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
    assert_eq!(cache.stats(), before);
    Ok(())
}

#[test]
fn cache_shares_direction_and_normalization_without_changing_execution() -> Result<(), SciOxideError>
{
    use scioxide_core::ExecutionContext;
    use scioxide_fft::{FftInput, FftOutput, HermitianSpectrum};
    for n in [1, 8, 17, 23, 210] {
        for layout in [FftLayout::Complex, FftLayout::Real, FftLayout::Hermitian] {
            let cx = ExecutionContext::sequential();
            let mut cache = FftPlanCache::new(FftCachePolicy::new(1, usize::MAX)?)?;
            let input = signal(n);
            let real: Vec<_> = input.iter().map(|z| z.re).collect();
            let packed = scioxide_fft::rfft(&real, Normalization::Unscaled)?;
            let mut bytes = None;
            for norm in [
                Normalization::Forward,
                Normalization::Inverse,
                Normalization::Orthonormal,
                Normalization::Unscaled,
            ] {
                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                    let spec = FftPlanSpec::new(n, direction, norm, layout);
                    let plan = cache.plan(spec, cx)?;
                    let fresh = FftPlan::build(spec, cx)?;
                    assert_eq!(plan.spec(), &spec);
                    assert_eq!(plan.plan_bytes(), fresh.plan_bytes());
                    assert_eq!(
                        plan.workspace_requirements(())?,
                        fresh.workspace_requirements(())?
                    );
                    let retained = cache.stats().retained_plan_bytes();
                    assert_eq!(*bytes.get_or_insert(retained), retained);
                    assert_eq!(plan.plan_bytes(), retained + core::mem::size_of_val(&plan));
                    assert_eq!(cache.stats().entries(), 1);
                    let mut work = plan.workspace(())?;
                    let result = |p: &FftPlan<f64>,
                                  work: &mut [Complex<f64>]|
                     -> Result<Vec<Complex<f64>>, SciOxideError> {
                        match (layout, direction) {
                            (FftLayout::Complex, _) => {
                                let mut out = input.clone();
                                p.execute_in_place(&mut out, work, cx)?;
                                Ok(out)
                            }
                            (FftLayout::Real, FftDirection::Forward)
                            | (FftLayout::Hermitian, FftDirection::Inverse) => {
                                let mut out = HermitianSpectrum::zeros(n)?;
                                p.execute_into(
                                    &real,
                                    FftOutput::Hermitian {
                                        logical_len: n,
                                        bins: scioxide_linalg::VectorViewMut::from_slice(
                                            out.bins_mut(),
                                        ),
                                    },
                                    work,
                                    cx,
                                )?;
                                Ok(out.bins().to_vec())
                            }
                            (FftLayout::Real, FftDirection::Inverse)
                            | (FftLayout::Hermitian, FftDirection::Forward) => {
                                let mut out = vec![0.0; n];
                                p.execute_into(
                                    FftInput::Hermitian {
                                        logical_len: n,
                                        bins: scioxide_linalg::VectorView::from_slice(
                                            packed.bins(),
                                        ),
                                    },
                                    &mut out,
                                    work,
                                    cx,
                                )?;
                                Ok(out.into_iter().map(|x| Complex::new(x, 0.0)).collect())
                            }
                        }
                    };
                    for (a, b) in result(&plan, &mut work)?
                        .iter()
                        .zip(result(&fresh, &mut work)?)
                    {
                        assert!((*a - b).norm() <= 8.0 * f64::EPSILON * (1.0 + b.norm()));
                    }
                }
            }
            assert_eq!(cache.stats().misses(), 1);
            assert_eq!(cache.stats().hits(), 7);
            assert_eq!(cache.stats().evictions(), 0);
        }
    }
    Ok(())
}

#[test]
fn cache_core_byte_limit_and_lifetime_are_independent_of_handles() -> Result<(), SciOxideError> {
    let cx = scioxide_core::ExecutionContext::sequential();
    let spec =
        |n, direction| FftPlanSpec::new(n, direction, Normalization::Inverse, FftLayout::Complex);
    let forward_spec = spec(8, FftDirection::Forward);
    let core_bytes = FftPlan::<f32>::build(forward_spec, cx)?.plan_bytes()
        - core::mem::size_of::<FftPlan<f32>>();
    let mut cache = FftPlanCache::new(FftCachePolicy::new(1, core_bytes)?)?;
    let forward = cache.plan(forward_spec, cx)?;
    let inverse = cache.plan(spec(8, FftDirection::Inverse), cx)?;
    assert_eq!(cache.stats().retained_plan_bytes(), core_bytes);
    let _ = cache.plan(spec(1, FftDirection::Forward), cx)?;
    assert_eq!(cache.stats().evictions(), 1);
    cache.clear();
    assert_eq!(cache.stats().entries(), 0);
    assert_eq!(cache.stats().retained_plan_bytes(), 0);
    let mut values = [Complex::new(1.0_f32, -0.5); 8];
    let mut scratch = forward.workspace(())?;
    forward.execute_in_place(&mut values, &mut scratch, cx)?;
    inverse.execute_in_place(&mut values, &mut scratch, cx)?;
    assert!(
        values
            .iter()
            .all(|z| (*z - Complex::new(1.0, -0.5)).norm() < 1e-6)
    );
    let mut tiny = FftPlanCache::<f32>::new(FftCachePolicy::new(1, core_bytes - 1)?)?;
    let _ = tiny.plan(forward_spec, cx)?;
    assert_eq!(tiny.stats().entries(), 0);
    Ok(())
}

#[test]
fn cache_keys_keep_layout_policy_and_resolved_backend_distinct() -> Result<(), SciOxideError> {
    let cx = scioxide_fft::FftPlanningContext::<f64>::default();
    let spec = FftPlanSpec::new(
        24,
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    );
    let mut cache = FftPlanCache::new(FftCachePolicy::new(16, usize::MAX)?)?;
    let first = cache.plan(spec, cx)?;
    let scioxide_fft::FftBackend::Cpu(first_backend) = first.resolved_backends().next().unwrap()
    else {
        unreachable!("CPU-only planning context")
    };
    let _ = cache.plan(
        spec.with_backend_strategy(FftBackendStrategy::Required(first_backend)),
        cx,
    )?;
    assert_eq!(cache.stats().hits(), 1);
    let _ = cache.plan(spec.with_execution_policy(ExecutionPolicy::Auto), cx)?;
    let _ = cache.plan(
        FftPlanSpec::new(
            24,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        ),
        cx,
    )?;
    let _ = cache.plan(
        FftPlanSpec::new(
            25,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        cx,
    )?;
    let mut expected = 4;
    for backend in KernelBackend::available(cx.execution().processor())
        // AVX-512 has no FFT leaves; requiring it is rejected.
        .filter(|b| *b != first_backend && *b != KernelBackend::Avx512)
    {
        let plan = cache.plan(
            spec.with_backend_strategy(FftBackendStrategy::Required(backend)),
            cx,
        )?;
        assert_eq!(
            plan.resolved_backends().next().unwrap(),
            scioxide_fft::FftBackend::Cpu(backend)
        );
        expected += 1;
    }
    assert_eq!(cache.stats().entries(), expected);
    assert_eq!(cache.stats().misses(), expected as u64);
    Ok(())
}
