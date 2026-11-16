//! Independent host proofs versus untrusted selection policy.
#![allow(clippy::unwrap_used)]

use scioxide_core::backend::{KernelBackend, KernelDispatch};
use scioxide_core::{
    InvalidInputReason, PortableProcessor, ProcessorCapabilities, ProcessorFeature,
    SciOxideErrorView,
};

struct Liar;
impl ProcessorCapabilities for Liar {
    fn supports(&self, _: ProcessorFeature) -> bool {
        true
    }
}

const BACKENDS: [KernelBackend; 4] = [
    KernelBackend::Scalar,
    KernelBackend::Avx2Fma,
    KernelBackend::Avx512,
    KernelBackend::Neon,
];

// Independent oracle: do not call SciOxide's availability or provider adapters.
fn proven(backend: KernelBackend) -> bool {
    match backend {
        KernelBackend::Scalar => true,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelBackend::Avx2Fma => {
            if cfg!(feature = "std") {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            } else {
                cfg!(target_feature = "avx2") && cfg!(target_feature = "fma")
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelBackend::Avx512 => {
            if cfg!(feature = "std") {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512vl")
            } else {
                cfg!(target_feature = "avx512f")
                    && cfg!(target_feature = "avx512dq")
                    && cfg!(target_feature = "avx512vl")
            }
        }
        #[cfg(target_arch = "aarch64")]
        KernelBackend::Neon => {
            if cfg!(feature = "std") {
                std::arch::is_aarch64_feature_detected!("neon")
            } else {
                cfg!(target_feature = "neon")
            }
        }
        _ => false,
    }
}

#[test]
fn lying_provider_cannot_enable_unverified_backends() {
    let expected: Vec<_> = BACKENDS.into_iter().filter(|&b| proven(b)).collect();
    eprintln!(
        "verified backends (std={}): {expected:?}",
        cfg!(feature = "std")
    );
    assert_eq!(
        KernelBackend::available(&Liar).collect::<Vec<_>>(),
        expected
    );
    let strongest = [
        KernelBackend::Avx512,
        KernelBackend::Avx2Fma,
        KernelBackend::Neon,
    ]
    .into_iter()
    .find(|&b| proven(b))
    .unwrap_or(KernelBackend::Scalar);
    {
        assert_eq!(KernelDispatch::detect(&Liar).backend(), strongest);
    };
    for backend in BACKENDS {
        assert_eq!(backend.is_available(&Liar), proven(backend));
        if proven(backend) {
            {
                assert_eq!(
                    KernelDispatch::for_backend(backend, &Liar)
                        .unwrap()
                        .backend(),
                    backend
                );
            };
        } else {
            let error = KernelDispatch::for_backend(backend, &Liar).unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
        }
    }
}

#[test]
fn every_required_feature_can_be_forbidden() {
    struct Except(ProcessorFeature);
    impl ProcessorCapabilities for Except {
        fn supports(&self, feature: ProcessorFeature) -> bool {
            feature != self.0
        }
    }
    for (backend, feature) in [
        (KernelBackend::Avx2Fma, ProcessorFeature::X86Avx2),
        (KernelBackend::Avx2Fma, ProcessorFeature::X86Fma),
        (KernelBackend::Avx512, ProcessorFeature::X86Avx512F),
        (KernelBackend::Avx512, ProcessorFeature::X86Avx512Dq),
        (KernelBackend::Avx512, ProcessorFeature::X86Avx512Vl),
        (KernelBackend::Neon, ProcessorFeature::Aarch64Neon),
    ] {
        assert!(!backend.is_available(&Except(feature)));
        assert!(KernelDispatch::for_backend(backend, &Except(feature)).is_err());
    }
    assert_eq!(
        KernelDispatch::detect(&PortableProcessor).backend(),
        KernelBackend::Scalar
    );
    assert_eq!(
        KernelBackend::available(&PortableProcessor).collect::<Vec<_>>(),
        [KernelBackend::Scalar]
    );
}
