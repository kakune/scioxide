//! Forward and inverse comparisons exercise every executable FFT backend.
#![expect(clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;

use scioxide_core::backend::KernelBackend;
use scioxide_core::{
    Complex, ExecutionContext, ProcessorCapabilities, ProcessorFeature, RealScalar,
};
use scioxide_fft::{
    FftBackendStrategy, FftDirection, FftLayout, FftPlan, FftPlanSpec, FftPlanningContext,
    Normalization,
};
use scioxide_linalg::{VectorView, VectorViewMut};

fn parity<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let context = FftPlanningContext::<S>::default();
    for n in [
        4, 16, 64, 256, 1024, 4096, 16_384, 65_536, 60, 125, 210, 1000, 2401, 3125, 3430, 4200,
        6720, 16_807,
    ] {
        // Offset slices exercise unaligned SIMD loads and stores.
        let input: Vec<_> = (0..n + 1)
            .map(|i| {
                Complex::new(
                    S::try_from_f64_lossy(((i * 73 % 193) as f64 - 96.0) / 128.0).unwrap(),
                    S::try_from_f64_lossy(((i * 107 % 193) as f64 - 96.0) / 128.0).unwrap(),
                )
            })
            .collect();
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for normalization in [Normalization::Unscaled, Normalization::Inverse] {
                let spec = FftPlanSpec::new(n, direction, normalization, FftLayout::Complex);
                let scalar = FftPlan::<S>::new(
                    spec.with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar)),
                )
                .unwrap();
                let mut expected = input[1..].to_vec();
                scalar
                    .execute_in_place(
                        &mut expected,
                        &mut scalar.workspace(()).unwrap(),
                        context.execution(),
                    )
                    .unwrap();
                for backend in numerics::fft_backends(context.execution().processor())
                    .filter(|b| *b != KernelBackend::Scalar)
                {
                    let plan = FftPlan::<S>::new(
                        spec.with_backend_strategy(FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let mut actual = vec![Complex::new(S::ZERO, S::ZERO); n + 1];
                    plan.execute_into(
                        VectorView::from_slice(&input[1..]),
                        VectorViewMut::from_slice(&mut actual[1..]),
                        &mut plan.workspace(()).unwrap(),
                        context.execution(),
                    )
                    .unwrap();
                    let (error, norm) = actual[1..].iter().zip(&expected).fold(
                        (0.0, 0.0),
                        |(error, norm), (a, e)| {
                            let dr = (a.re - e.re).try_to_f64_exact().unwrap();
                            let di = (a.im - e.im).try_to_f64_exact().unwrap();
                            let re = e.re.try_to_f64_exact().unwrap();
                            let im = e.im.try_to_f64_exact().unwrap();
                            (error + dr * dr + di * di, norm + re * re + im * im)
                        },
                    );
                    let bound = 2.0 * S::EPSILON.try_to_f64_exact().unwrap() * (n as f64).log2();
                    numerics::assert_close(
                        (error / norm).sqrt(),
                        bound,
                        format_args!("{backend:?} {direction:?} {normalization:?} n={n}"),
                    );
                }
            }
        }
    }
}

#[test]
fn all_backends_match_scalar_f32() {
    parity::<f32>();
}

#[test]
fn all_backends_match_scalar_f64() {
    parity::<f64>();
}

struct Policy {
    avx2: bool,
    avx512: bool,
    neon: bool,
}

impl ProcessorCapabilities for Policy {
    fn supports(&self, feature: ProcessorFeature) -> bool {
        match feature {
            ProcessorFeature::X86Avx2 | ProcessorFeature::X86Fma => self.avx2,
            ProcessorFeature::X86Avx512F
            | ProcessorFeature::X86Avx512Dq
            | ProcessorFeature::X86Avx512Vl => self.avx512,
            ProcessorFeature::Aarch64Neon => self.neon,
            _ => false,
        }
    }
}

#[test]
fn automatic_uses_fft_ranking_and_respects_processor_policy() {
    for policy in [
        Policy {
            avx2: true,
            avx512: true,
            neon: true,
        },
        Policy {
            avx2: false,
            avx512: true,
            neon: true,
        },
        Policy {
            avx2: true,
            avx512: false,
            neon: false,
        },
        Policy {
            avx2: false,
            avx512: false,
            neon: false,
        },
    ] {
        let context = ExecutionContext::sequential().with_processor(&policy);
        // AVX-512 has no FFT leaves, so its capability never changes the
        // automatic choice (the AVX-512-only policy falls through to NEON or
        // scalar).
        let expected = [KernelBackend::Avx2Fma, KernelBackend::Neon]
            .into_iter()
            .find(|backend| backend.is_available(&policy))
            .unwrap_or(KernelBackend::Scalar);
        for layout in [FftLayout::Complex, FftLayout::Real] {
            let plan = FftPlan::<f64>::build(
                FftPlanSpec::new(1024, FftDirection::Forward, Normalization::Inverse, layout),
                context,
            )
            .unwrap();
            assert_eq!(
                plan.resolved_backends().next().unwrap(),
                scioxide_fft::FftBackend::Cpu(expected)
            );
        }
    }
}
