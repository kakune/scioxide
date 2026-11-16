//! Checked one-dimensional real FFT execution.
use super::*;

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Transform real samples into a packed-spectrum view, conjugating the
    /// written bins when `conjugate` is set (the Hermitian inverse).
    ///
    /// Contiguous input is packed with one tight loop and contiguous output
    /// receives the even split fused with restoration; strided views take
    /// separate iterator paths with identical arithmetic.
    ///
    /// # Errors
    ///
    /// Returns typed layout, shape, workspace, non-finite, cancellation,
    /// deadline, execution, or numerical failures. Output is unchanged on
    /// failure.
    pub(crate) fn forward_into(
        &self,
        scale: S,
        input: VectorView<'_, S>,
        mut output: VectorViewMut<'_, Complex<S>>,
        conjugate: bool,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        S: scioxide_core::NumericScalar<Real = S>,
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        crate::validation::length(input.len(), self.n)?;
        crate::validation::length(output.len(), self.spectrum_len)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        context.check()?;
        let exponent = match input.as_contiguous_slice() {
            Some(samples) => self.forward_slice(samples, scratch, context)?,
            None => self.forward_scaled(|| input.iter(), scratch, context)?,
        };
        context.check()?;
        let sign = if conjugate { -S::ONE } else { S::ONE };
        let bins = match output.as_contiguous_slice_mut() {
            Some(bins) => Bins::Contiguous(bins),
            None => Bins::Strided(output),
        };
        self.publish_bins(scale, sign, scratch, exponent, bins)
    }

    /// Measure and stage range-scaled contiguous `samples`, then run the
    /// child. Returns the restoring exponent, or `None` for an all-zero signal.
    pub(super) fn forward_slice(
        &self,
        samples: &[S],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<Option<i32>, SciOxideError> {
        let exponent = self.stage_scaled(
            scratch,
            crate::scaling::real_max_slice(samples)?,
            |data, down| {
                self.stage_slice(data, samples, down);
            },
        );
        if exponent.is_some() {
            self.transform_staged(scratch, context)?;
        }
        Ok(exponent)
    }

    /// [`Self::forward_slice`] for samples in any order.
    pub(super) fn forward_scaled<I: Iterator<Item = S>>(
        &self,
        samples: impl Fn() -> I,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<Option<i32>, SciOxideError> {
        let exponent = self.stage_scaled(scratch, real_max(samples())?, |data, down| {
            self.stage_samples(data, samples(), down);
        });
        if exponent.is_some() {
            self.transform_staged(scratch, context)?;
        }
        Ok(exponent)
    }

    /// Bounded child for a parent that owns scaling, checks and publication.
    /// `stage` fills the staging slots (see [`Self::stage_samples`]) with
    /// finite samples <=1; the unscaled split spectrum is returned in scratch.
    pub(crate) fn forward_bounded<'a>(
        &self,
        scratch: &'a mut [Complex<S>],
        stage: impl FnOnce(&mut [Complex<S>]),
        context: ExecutionContext<'_>,
    ) -> Result<&'a [Complex<S>], SciOxideError> {
        debug_assert!(scratch.len() >= self.scratch_len);
        stage(&mut scratch[..self.staged_len()]);
        self.transform_staged(scratch, context)?;
        if let RealAlgorithm::Even { twiddles, .. } = &self.algorithm {
            postprocess_even_forward(scratch, twiddles, self.n / 2);
        }
        Ok(&scratch[..self.spectrum_len])
    }

    /// Reconstruct real samples from a packed-spectrum view, conjugating the
    /// staged bins when `conjugate` is set (the Hermitian forward). The
    /// Hermitian input policy and finite checks are conjugation-invariant.
    ///
    /// # Errors
    ///
    /// Returns typed layout, shape, workspace, endpoint, non-finite,
    /// cancellation, deadline, execution, or numerical failures. Output is
    /// unchanged on failure.
    pub(crate) fn inverse_into(
        &self,
        scale: S,
        input: VectorView<'_, Complex<S>>,
        mut output: VectorViewMut<'_, S>,
        conjugate: bool,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        S: scioxide_core::NumericScalar<Real = S>,
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        crate::validation::length(input.len(), self.spectrum_len)?;
        crate::validation::length(output.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        context.check()?;
        let sign = if conjugate { -S::ONE } else { S::ONE };
        let (factor, restore) = match input.as_contiguous_slice() {
            Some(bins) => self.inverse_bins(scale, bins.iter().copied(), sign, scratch, context),
            None => self.inverse_bins(scale, input.iter(), sign, scratch, context),
        }?;
        context.check()?;
        let samples = &scratch[..self.staged_len()];
        let publish = |value: S| restore.restore(value * factor);
        match output.as_contiguous_slice_mut() {
            Some(output) if self.is_packed() => {
                let mut pairs = output.chunks_exact_mut(2);
                for (pair, value) in (&mut pairs).zip(samples) {
                    pair[0] = publish(value.re);
                    pair[1] = publish(value.im);
                }
                if let [last] = pairs.into_remainder() {
                    *last = publish(samples[self.n / 2].re);
                }
            }
            Some(output) => self.publish_samples(samples, factor, restore, output.iter_mut()),
            None => self.publish_samples(samples, factor, restore, output.iter_mut()),
        }
        Ok(())
    }

    /// Stage packed `bins` with imaginary parts times `sign`, project the
    /// endpoints with the spectrum maximum and invert. The real samples are
    /// left in the staging slots with their publication factor and restoration.
    pub(super) fn inverse_bins(
        &self,
        scale: S,
        bins: impl Iterator<Item = Complex<S>>,
        sign: S,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(S, BinaryScale<S>), SciOxideError> {
        let packed = &mut scratch[..self.spectrum_len];
        let maximum = stage_bins(packed, bins, sign)?;
        packed[0] = crate::hermitian::project_pair(packed[0], None, maximum)?;
        if self.n.is_multiple_of(2) {
            packed[self.n / 2] = crate::hermitian::project_pair(packed[self.n / 2], None, maximum)?;
        }
        self.inverse_core(scale, scratch, context, range_exponent(maximum) + 2)
    }

    /// Invert the projected spectrum staged in scratch. The real samples are
    /// left in the staging slots, ready for publication with the returned
    /// factor and restoration; overflow is detected before returning.
    pub(super) fn inverse_core(
        &self,
        scale: S,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
        input_exponent: i32,
    ) -> Result<(S, BinaryScale<S>), SciOxideError> {
        let scaling = BinaryScale::new(-input_exponent);
        let prepared = match &self.algorithm {
            RealAlgorithm::Singleton => {
                scratch[0].re = checked_restore(scratch[0].re, 0, scale)?;
                (S::ONE, BinaryScale::Identity)
            }
            RealAlgorithm::Even { complex, twiddles } => {
                let (data, child_scratch) = scratch.split_at_mut(self.spectrum_len);
                preprocess_even_inverse(data, twiddles, self.n / 2, scaling);
                complex.execute_bounded(
                    &mut data[..self.n / 2],
                    child_scratch,
                    true,
                    S::ONE,
                    context,
                )?;
                // The length-m inverse yields m*z; the length-2m real inverse
                // needs 2m*x. Keep this binary factor separate from normalization.
                let bound = self.output_bound(scale);
                match output_factor(bound, input_exponent + 1, scale) {
                    Some(prepared) => prepared,
                    None => {
                        restore_checked(&mut data[..self.n / 2], input_exponent + 1, scale, bound)?
                    }
                }
            }
            RealAlgorithm::Decimated(_) | RealAlgorithm::Odd { .. } => {
                self.inverse_odd_raw(scratch, scaling, context)?;
                let factor = scale * self.gain();
                let restore = BinaryScale::new(input_exponent);
                if self
                    .output_bound(scale)
                    .scalbn(input_exponent + 2)
                    .is_finite()
                {
                    (factor, restore)
                } else {
                    let packed = self.is_packed();
                    let checked = |value: S| {
                        let value = restore.restore(value * factor);
                        if value.is_finite() {
                            Ok(value)
                        } else {
                            Err(crate::validation::real_transform_overflow())
                        }
                    };
                    for slot in &mut scratch[..self.staged_len()] {
                        slot.re = checked(slot.re)?;
                        if packed {
                            slot.im = checked(slot.im)?;
                        }
                    }
                    (S::ONE, BinaryScale::Identity)
                }
            }
        };
        Ok(prepared)
    }

    /// Factor between an odd plan's raw inverse and `n` times its samples.
    pub(super) fn gain(&self) -> S {
        match &self.algorithm {
            RealAlgorithm::Decimated(decimation) => decimation.gain(),
            _ => S::ONE,
        }
    }

    /// Unnormalized odd-length inverse of the packed spectrum in scratch,
    /// whose components scaled by `scaling` are finite and at most 1/4. The
    /// staging slots receive `n / gain` times the real samples. Promoted
    /// plans check the imaginary residual of their complex inverse;
    /// decimated plans are real by construction.
    pub(super) fn inverse_odd_raw(
        &self,
        scratch: &mut [Complex<S>],
        scaling: BinaryScale<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (data, work) = self.split_staging(scratch);
        match &self.algorithm {
            RealAlgorithm::Decimated(decimation) => {
                decimation.inverse(data, work, scaling, context)
            }
            RealAlgorithm::Odd { complex } => {
                data[0] = Complex::new(scaling.apply(data[0].re), S::ZERO);
                for index in 1..=self.n / 2 {
                    let value = data[index];
                    let value = Complex::new(scaling.apply(value.re), scaling.apply(value.im));
                    data[index] = value;
                    data[self.n - index] = value.conj();
                }
                let mass = crate::hermitian::component_mass(data.iter().copied());
                complex.execute_bounded(data, work, true, S::ONE, context)?;
                check_odd_inverse_residual(data, self.n, mass)
            }
            RealAlgorithm::Singleton | RealAlgorithm::Even { .. } => {
                unreachable!("only odd lengths above one have an odd raw inverse")
            }
        }
    }
}
