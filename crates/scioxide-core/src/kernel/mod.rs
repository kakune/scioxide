//! Selection of CPU-specific kernels that the host is verified to support.

use crate::{InvalidInputReason, ProcessorCapabilities, ProcessorFeature, SciOxideError};

/// Instruction set used by a numerical kernel.
///
/// A value of this enum is only a request. [`KernelDispatch`] confirms that the
/// host supports it. `Scalar` is always available.
///
/// ```
/// use scioxide_core::{PortableProcessor, backend::KernelBackend};
/// assert_eq!(
///     KernelBackend::available(&PortableProcessor).collect::<Vec<_>>(),
///     [KernelBackend::Scalar]
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum KernelBackend {
    /// Portable scalar code.
    Scalar,
    /// x86/x86-64 AVX2 with fused multiply-add.
    Avx2Fma,
    /// x86/x86-64 AVX-512F, AVX-512DQ, and AVX-512VL.
    Avx512,
    /// AArch64 NEON.
    Neon,
}

impl KernelBackend {
    /// Returns whether the host supports this backend and `processor` allows it.
    ///
    /// With `std` the host is checked at run time; without it, only compile-time
    /// target features count. `processor` can forbid a backend but cannot
    /// enable one the host check did not confirm.
    pub fn is_available(self, processor: &dyn ProcessorCapabilities) -> bool {
        let permitted =
            |feature: ProcessorFeature| feature.verified() && processor.supports(feature);
        match self {
            Self::Scalar => true,
            Self::Avx2Fma => {
                permitted(ProcessorFeature::X86Avx2) && permitted(ProcessorFeature::X86Fma)
            }
            Self::Avx512 => {
                permitted(ProcessorFeature::X86Avx512F)
                    && permitted(ProcessorFeature::X86Avx512Dq)
                    && permitted(ProcessorFeature::X86Avx512Vl)
            }
            Self::Neon => permitted(ProcessorFeature::Aarch64Neon),
        }
    }

    /// Returns every backend for which [`Self::is_available`] holds, from
    /// `Scalar` to the most specialized one.
    pub fn available(processor: &dyn ProcessorCapabilities) -> impl Iterator<Item = Self> + '_ {
        [Self::Scalar, Self::Avx2Fma, Self::Avx512, Self::Neon]
            .into_iter()
            .filter(|backend| backend.is_available(processor))
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2Fma => "avx2-fma",
            Self::Avx512 => "avx512f-dq-vl",
            Self::Neon => "neon",
        }
    }
}

/// A kernel backend that the host is verified to support.
///
/// It can only be created by [`Self::detect`] or [`Self::for_backend`], which
/// check the host as described in [`KernelBackend::is_available`]. Kernels that
/// receive a `KernelDispatch` can therefore use its instruction set safely.
///
/// ```
/// use scioxide_core::{PortableProcessor, backend::{KernelBackend, KernelDispatch}};
/// let token = KernelDispatch::for_backend(KernelBackend::Scalar, &PortableProcessor)?;
/// assert_eq!(token.backend(), KernelBackend::Scalar);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct KernelDispatch {
    backend: KernelBackend,
}

impl KernelDispatch {
    /// Selects the most specialized available backend: AVX-512, then AVX2 with
    /// FMA, then NEON, then `Scalar`.
    pub fn detect(processor: &dyn ProcessorCapabilities) -> Self {
        // Strongest complete feature set first. Each kernel owns its arithmetic
        // and rounding contract; the token proves only the selected ISA.
        let backend = [
            KernelBackend::Avx512,
            KernelBackend::Avx2Fma,
            KernelBackend::Neon,
        ]
        .into_iter()
        .find(|backend| backend.is_available(processor))
        .unwrap_or(KernelBackend::Scalar);
        Self { backend }
    }

    /// Selects `backend` after checking that it is available.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` ([`InvalidInputReason::UnsupportedConfiguration`]) if the
    ///   host does not support `backend` or `processor` forbids it.
    pub fn for_backend(
        backend: KernelBackend,
        processor: &dyn ProcessorCapabilities,
    ) -> Result<Self, SciOxideError> {
        if !backend.is_available(processor) {
            return Err(SciOxideError::invalid_input(
                crate::argument!("backend"),
                InvalidInputReason::UnsupportedConfiguration,
                alloc::format!("kernel backend {} is unavailable", backend.name()),
            ));
        }
        Ok(Self { backend })
    }

    /// Returns the selected backend.
    pub const fn backend(self) -> KernelBackend {
        self.backend
    }
}
