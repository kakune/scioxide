//! Real FFT preparation and resource sizing.
use super::*;

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Create a real FFT plan with explicit execution and processor
    /// capabilities.
    ///
    /// A rejected length is attributed to `length` and provider failures to `provider`, the caller
    /// parameters that determined the length and carried the context.
    ///
    /// # Errors
    ///
    /// Returns typed size, allocation, or numerical planning errors.
    pub(crate) fn with_context<'a>(
        n: usize,
        execution_policy: ExecutionPolicy,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
        length: Argument,
        provider: Argument,
    ) -> Result<Self, SciOxideError> {
        let context = context.into();
        Self::with_backend_and_context(
            n,
            execution_policy,
            context.resolve(crate::FftBackendStrategy::Automatic)?,
            context,
            length,
            provider,
        )
    }

    /// Like [`Self::with_context`], with a required CPU `backend`.
    pub(crate) fn with_backend_and_context<'a>(
        n: usize,
        execution_policy: ExecutionPolicy,
        backend: KernelBackend,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
        length: Argument,
        provider: Argument,
    ) -> Result<Self, SciOxideError> {
        let context = context.into();
        crate::validation::nonzero(n, length)?;
        let growth = scalar_from_usize_rounded::<S>(n)? * <S as scioxide_core::RealScalar>::TWO;
        let backend = context.resolve(crate::FftBackendStrategy::Required(backend))?;
        let spectrum_len = checked_add(
            n / 2,
            1,
            scioxide_core::quantity!("real_fft_spectrum_length"),
        )?;

        let (algorithm, backend, owned_bytes, diagnostics) = prepare_algorithm(
            n,
            spectrum_len,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        // Even plans stage m + 1 values (the packed signal plus the extra
        // Nyquist slot written by the split) and decimated plans their packed
        // signal, which the spectrum then replaces; promoted odd plans stage
        // the full complex signal. Child scratch follows the staging area.
        let q = scioxide_core::quantity!("real_fft_scratch_length");
        let scratch_len = match &algorithm {
            RealAlgorithm::Singleton => 1,
            RealAlgorithm::Even { complex, .. } => {
                checked_add(spectrum_len, complex.bounded_scratch_len(), q)?
            }
            RealAlgorithm::Decimated(decimation) => {
                checked_add(spectrum_len, decimation.work_len()?, q)?
            }
            RealAlgorithm::Odd { complex } => checked_add(n, complex.bounded_scratch_len(), q)?,
        };
        let stages = match &algorithm {
            RealAlgorithm::Even { complex, .. } if n / 2 <= ROWS => complex.smooth_stages(),
            _ => None,
        };
        let blocks = stages
            .map(|_| crate::backend::dispatch(backend, context.execution().processor()))
            .transpose()?;
        let digit_rows = stages.map_or(Ok(Vec::new()), crate::kernel::StagePlan::digit_rows)?;
        let plan_bytes = checked_add(
            size_of::<Self>(),
            checked_add(
                owned_bytes,
                crate::memory::vector_bytes(&digit_rows),
                scioxide_core::quantity!("real_fft_resident_plan_bytes"),
            )?,
            scioxide_core::quantity!("real_fft_resident_plan_bytes"),
        )?;
        let provider = match &algorithm {
            RealAlgorithm::Singleton => false,
            RealAlgorithm::Even { complex, .. } | RealAlgorithm::Odd { complex } => {
                complex.resolved_backend() == crate::FftBackend::Provider
            }
            RealAlgorithm::Decimated(decimation) => {
                let (pair, residue) = decimation.children();
                pair.resolved_backend() == crate::FftBackend::Provider
                    || residue.growth.is_infinite()
            }
        };
        Ok(Self {
            n,
            spectrum_len,
            growth: if provider { S::INFINITY } else { growth },
            backend,
            scratch_len,
            plan_bytes,
            diagnostics,
            algorithm,
            blocks,
            digit_rows,
        })
    }
}

/// Select the real-data factorization and keep its child resources.
fn prepare_algorithm<S: scioxide_core::RealScalar>(
    n: usize,
    spectrum_len: usize,
    execution_policy: ExecutionPolicy,
    backend: KernelBackend,
    context: crate::FftPlanningContext<'_, S>,
    length: Argument,
    provider: Argument,
) -> Result<(RealAlgorithm<S>, KernelBackend, usize, FftPlanDiagnostics), SciOxideError> {
    Ok(if n == 1 {
        let probe: ComplexCore<S> = ComplexCore::with_backend_and_context(
            1,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        (
            RealAlgorithm::Singleton,
            probe.backend(),
            0,
            probe.diagnostics(),
        )
    } else if n.is_multiple_of(2) {
        // Even n = 2m packs z[j] = x[2j] + i x[2j+1] into one length-m
        // complex FFT and separates the spectra afterwards with
        // w^k = exp(-2*pi*i*k/n), k = 0..=m. The twiddles use the same
        // independently evaluated roots as complex stages; the split
        // step is reported as one more stage.
        let half = n / 2;
        let complex = ComplexCore::with_backend_and_context(
            half,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        let mut twiddles =
            try_with_capacity(spectrum_len, scioxide_core::quantity!("real_fft_twiddles"))?;
        for index in 0..spectrum_len {
            twiddles.push(reduced_root::<S>(index as u128, n as u128, false, length)?);
        }
        let twiddle_bytes = checked_mul(
            twiddles.capacity(),
            size_of::<Complex<S>>(),
            scioxide_core::quantity!("real_fft_twiddle_bytes"),
        )?;
        let owned_bytes = checked_add(
            complex.plan_bytes() - size_of::<ComplexCore<S>>(),
            twiddle_bytes,
            scioxide_core::quantity!("real_fft_owned_plan_bytes"),
        )?;
        let backend = complex.backend();
        let diagnostics = complex.diagnostics().with_additional_stage();
        (
            RealAlgorithm::Even { complex, twiddles },
            backend,
            owned_bytes,
            diagnostics,
        )
    } else if let Some(radix) = decimation::radix(n) {
        let decimation = Decimation::new(
            n,
            radix,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        let (backend, owned_bytes, diagnostics) = (
            decimation.backend(),
            decimation.owned_bytes()?,
            decimation.diagnostics(),
        );
        (
            RealAlgorithm::Decimated(decimation),
            backend,
            owned_bytes,
            diagnostics,
        )
    } else {
        // Other odd lengths have no real-data factorization here, so the
        // real signal is promoted to a full-length complex transform.
        let complex = ComplexCore::with_backend_and_context(
            n,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        let owned_bytes = complex.plan_bytes() - size_of::<ComplexCore<S>>();
        let backend = complex.backend();
        let diagnostics = complex.diagnostics();
        (
            RealAlgorithm::Odd { complex },
            backend,
            owned_bytes,
            diagnostics,
        )
    })
}
