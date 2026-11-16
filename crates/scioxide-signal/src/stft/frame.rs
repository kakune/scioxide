//! Forward-only window framing shared by STFT and spectral estimation.
use super::*;
use scioxide_core::backend::{components_mut, standard};
use scioxide_fft::FftOutput;

#[derive(Clone, Debug)]
pub(crate) struct FramePlan<T: NumericScalar> {
    pub(super) window: Vec<T>,
    pub(super) options: StftOptions<T::Real>,
    /// Samples between the starts of consecutive frames.
    pub(crate) hop: usize,
    /// FFT length, resolved from the options and the window length.
    pub(crate) fft_len: usize,
    forward: FftPlan<T::Real>,
    pub(super) scale: T::Real,
    weights: Vec<T::Real>,
    window_exponent: i32,
    scale_exponent: i32,
    pub(crate) work_len: usize,
}
impl<T: NumericScalar> FramePlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    pub(crate) fn new(
        window: &[T],
        hop: usize,
        options: StftOptions<T::Real>,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if window.is_empty() {
            return Err(empty(
                scioxide_core::argument!("window"),
                "window must be nonempty",
            ));
        }
        if hop == 0 || hop > i64::MAX as usize {
            return Err(invalid(
                scioxide_core::argument!("hop"),
                "hop must be positive and fit a signed frame index",
            ));
        }
        let fft_len = options.fft_len.unwrap_or(window.len());
        if fft_len < window.len() {
            return Err(inconsistent(
                scioxide_core::argument!("options.fft_len"),
                "FFT length must cover the window",
            ));
        }
        if fft_len > i64::MAX as usize {
            return Err(invalid(
                scioxide_core::argument!("options.fft_len"),
                "FFT length must fit a signed index",
            ));
        }
        crate::util::check_real_slice_finite(&[options.sample_rate])?;
        if options.sample_rate <= T::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.sample_rate"),
                "sample rate must be positive",
            ));
        }
        finite(window)?;
        let complex_domain =
            T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() != T::Real::ZERO;
        if options.packing == StftPacking::OneSided && complex_domain {
            return Err(unsupported(
                scioxide_core::argument!("options.packing"),
                "one-sided packing requires real samples",
            ));
        }
        let scale = match options.scaling {
            StftScaling::Raw => T::Real::ONE,
            StftScaling::Magnitude => {
                let sum = crate::util::sum_products(
                    window
                        .iter()
                        .map(|&w| (complex(w), Complex::new(T::Real::ONE, T::Real::ZERO))),
                )?;
                if sum.norm() == T::Real::ZERO {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("window"),
                        scioxide_core::DomainReason::Undefined,
                        "magnitude scaling is undefined for a zero-sum window",
                    ));
                }
                T::Real::ONE / sum.norm()
            }
            StftScaling::Psd => {
                let norm = window
                    .iter()
                    .fold(T::Real::ZERO, |s, &w| s.hypot(complex(w).norm()));
                if norm == T::Real::ZERO {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("window"),
                        scioxide_core::DomainReason::Undefined,
                        "density scaling is undefined for an all-zero window",
                    ));
                }
                (T::Real::ONE / norm) / options.sample_rate.sqrt()
            }
        };
        if !scale.is_finite() || scale <= T::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                if scale.is_finite() {
                    scioxide_core::NumericalIssue::Underflow
                } else {
                    scioxide_core::NumericalIssue::Overflow
                },
                "window normalization for the requested scaling is not representable",
            ));
        }

        let forward = FftPlan::build(
            FftPlanSpec::new(
                fft_len,
                FftDirection::Forward,
                Normalization::Inverse,
                if T::IS_REAL {
                    FftLayout::Real
                } else {
                    FftLayout::Complex
                },
            )
            .with_execution_policy(options.execution_policy),
            context,
        )?;
        let work_len = checked_add(
            checked_mul(2, fft_len, scioxide_core::quantity!("frame_work"))?,
            forward.workspace_requirements(())?.complex_len(),
            scioxide_core::quantity!("frame_work"),
        )?;
        let bins = if options.packing == StftPacking::OneSided {
            fft_len / 2 + 1
        } else {
            fft_len
        };
        let mut weights = try_filled(
            bins,
            T::Real::ONE,
            scioxide_core::quantity!("frame_weights"),
        )?;
        if options.packing == StftPacking::OneSided && options.scaling == StftScaling::Psd {
            let root_two = T::Real::TWO.sqrt();
            for (k, weight) in weights.iter_mut().enumerate().skip(1) {
                if !(fft_len.is_multiple_of(2) && k == fft_len / 2) {
                    *weight = root_two;
                }
            }
        }
        Ok(Self {
            window: copy(window)?,
            options,
            hop,
            fft_len,
            forward,
            scale,
            weights,
            window_exponent: crate::util::fft_exponent(crate::util::scale(window), fft_len)?,
            scale_exponent: crate::util::fft_exponent(scale, fft_len)?,
            work_len,
        })
    }
    /// Number of frequency bins per frame.
    pub(crate) fn bins(&self) -> usize {
        if self.options.packing == StftPacking::OneSided {
            self.fft_len / 2 + 1
        } else {
            self.fft_len
        }
    }
    /// Frequency of an output bin, in sample-rate units.
    /// # Errors
    /// Rejects out-of-range bins or unrepresentable metadata.
    pub(crate) fn frequency(&self, bin: usize) -> Result<T::Real, SciOxideError> {
        if bin >= self.bins() {
            return Err(invalid(
                scioxide_core::argument!("bin"),
                "bin must be below the bin count",
            ));
        }
        let k = self.bin_index(bin);
        let signed =
            if self.options.packing != StftPacking::OneSided && k >= self.fft_len.div_ceil(2) {
                k as i128 - self.fft_len as i128
            } else {
                k as i128
            };
        let f = scalar_from_usize_rounded::<T::Real>(signed.unsigned_abs() as usize)?
            / scalar_from_usize_rounded::<T::Real>(self.fft_len)?
            * self.options.sample_rate;
        crate::util::finite(if signed < 0 { -f } else { f })
    }
    pub(super) fn bin_index(&self, bin: usize) -> usize {
        if self.options.packing == StftPacking::Centered {
            (bin + self.fft_len.div_ceil(2)) % self.fft_len
        } else {
            bin
        }
    }

    pub(super) fn bin_weight(&self, bin: usize) -> T::Real {
        self.weights[bin]
    }

    pub(crate) fn analyze_frame(
        &self,
        sample: impl Fn(usize) -> T,
        output: &mut [Complex<T::Real>],
        work: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let exponent = self.analyze_frame_scaled(sample, output, work, context)?;
        crate::util::restore_fft(output, exponent)
    }

    /// Binary exponent of the window and analysis-scale frames, shared by
    /// every frame; [`Self::analyze_frame_scaled`] adds the frame's own
    /// sample exponent to it.
    pub(crate) fn plan_exponent(&self) -> i32 {
        self.window_exponent + self.scale_exponent
    }

    /// Keep the common binary exponent until the consumer's final product.
    pub(crate) fn analyze_frame_scaled(
        &self,
        sample: impl Fn(usize) -> T,
        output: &mut [Complex<T::Real>],
        work: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<i32, SciOxideError> {
        let n = self.fft_len;
        let maximum = (0..self.window.len())
            .fold(T::Real::ZERO, |a, j| a.max(standard::magnitude(sample(j))));
        let exponent = crate::util::fft_exponent(maximum, n)?;
        let window_exponent = self.window_exponent;
        let (buffer, rest) = work.split_at_mut(n);
        let (spectrum, fft) = rest.split_at_mut(n);
        if T::IS_REAL {
            let values = &mut components_mut(buffer)[..n];
            values.fill(T::Real::ZERO);
            for (j, &w) in self.window.iter().enumerate() {
                values[j] = standard::scale(sample(j), -exponent).real_component()
                    * standard::scale(w, -window_exponent).real_component();
            }
            self.forward.execute_into(
                values,
                FftOutput::hermitian(n, &mut spectrum[..n / 2 + 1]),
                fft,
                context,
            )?;
        } else {
            buffer.fill(zero());
            for (j, &w) in self.window.iter().enumerate() {
                buffer[j] = complex(standard::scale(sample(j), -exponent))
                    * complex(standard::scale(w, -window_exponent)).conj();
            }
            self.forward
                .execute_into(&*buffer, &mut *spectrum, fft, context)?;
        }
        for (k, slot) in output.iter_mut().enumerate() {
            let index = self.bin_index(k);
            let z = if T::IS_REAL && index > n / 2 {
                spectrum[n - index].conj()
            } else {
                spectrum[index]
            };
            let factor = standard::scale(self.scale, -self.scale_exponent) * self.weights[k];
            *slot = z.scale(factor);
        }
        Ok(exponent + window_exponent + self.scale_exponent)
    }
}

/// Weighted product `u*conj(v)*weight` of two frame spectra, restored by their
/// combined binary `exponent`. Frames have product headroom, so the native
/// product only meets gradual underflow; a non-finite candidate replays exact
/// triple products.
pub(crate) fn weighted_product<S: RealScalar>(
    u: Complex<S>,
    v: Complex<S>,
    weight: S,
    exponent: i32,
) -> Result<Complex<S>, SciOxideError> {
    let z = standard::scale((u * v.conj()).scale(weight), exponent);
    if z.has_finite_components() {
        return Ok(z);
    }
    let mut re = scioxide_core::RangeAccumulator::new();
    let mut im = scioxide_core::RangeAccumulator::new();
    re.add_triple_product(u.re, v.re, weight)?;
    re.add_triple_product(u.im, v.im, weight)?;
    im.add_triple_product(u.im, v.re, weight)?;
    im.subtract_triple_product(u.re, v.im, weight)?;
    let finish = |accumulator: scioxide_core::RangeAccumulator<S>| {
        if exponent == 0 {
            accumulator.finish()
        } else {
            accumulator
                .finish_scaled::<S>()?
                .shifted(i64::from(exponent))?
                .round()
        }
    };
    Ok(Complex::new(finish(re)?, finish(im)?))
}
