//! Real samples and packed-bin staging.
use super::*;

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Staged input: `ceil(n / 2)` packed slots, or `n` real-valued slots.
    pub(super) const fn staged_len(&self) -> usize {
        if self.is_packed() {
            self.n.div_ceil(2)
        } else {
            self.n
        }
    }

    /// Staging storage at the front of `scratch` (with the even split's extra
    /// Nyquist slot) and the child's own scratch after it.
    pub(crate) fn split_staging<'a>(
        &self,
        scratch: &'a mut [Complex<S>],
    ) -> (&'a mut [Complex<S>], &'a mut [Complex<S>]) {
        scratch[..self.scratch_len].split_at_mut(if self.is_packed() {
            self.spectrum_len
        } else {
            self.n
        })
    }

    /// Stage samples `0..n` in order, applying `down` to each. Packed plans
    /// pair consecutive samples in one slot (an odd final sample takes a zero
    /// partner); promoted odd plans use real slots.
    pub(crate) fn stage_samples(
        &self,
        data: &mut [Complex<S>],
        samples: impl Iterator<Item = S>,
        down: BinaryScale<S>,
    ) {
        let mut samples = samples.map(|value| down.apply(value));
        if self.is_packed() {
            for slot in &mut data[..self.staged_len()] {
                let Some(re) = samples.next() else {
                    break;
                };
                *slot = Complex::new(re, samples.next().unwrap_or(S::ZERO));
            }
        } else {
            for (slot, re) in data[..self.n].iter_mut().zip(samples) {
                *slot = Complex::new(re, S::ZERO);
            }
        }
    }

    /// Write a contiguous run of mapped scalar samples, pairing packed slots
    /// without an index/parity branch for each sample. Runs may meet mid-slot.
    pub(crate) fn stage_run(
        &self,
        data: &mut [Complex<S>],
        first: usize,
        count: usize,
        sample: impl Fn(usize) -> S,
    ) {
        if self.is_packed() {
            let mut j = 0;
            if first % 2 == 1 && count != 0 {
                data[first / 2].im = sample(0);
                j = 1;
            }
            let start = (first + j) / 2;
            let pairs = (count - j) / 2;
            for (i, slot) in data[start..start + pairs].iter_mut().enumerate() {
                *slot = Complex::new(sample(j + 2 * i), sample(j + 2 * i + 1));
            }
            j += 2 * pairs;
            if j < count {
                data[(first + j) / 2] = Complex::new(sample(j), S::ZERO);
            }
        } else {
            for (j, slot) in data[first..first + count].iter_mut().enumerate() {
                *slot = Complex::new(sample(j), S::ZERO);
            }
        }
    }

    /// Contiguous [`Self::stage_samples`].
    pub(super) fn stage_slice(&self, data: &mut [Complex<S>], samples: &[S], down: BinaryScale<S>) {
        if self.is_packed() {
            let pairs = samples.chunks_exact(2);
            if let [last] = pairs.remainder() {
                data[self.n / 2] = Complex::new(down.apply(*last), S::ZERO);
            }
            for (slot, pair) in data.iter_mut().zip(pairs) {
                *slot = Complex::new(down.apply(pair[0]), down.apply(pair[1]));
            }
        } else {
            self.stage_samples(data, samples.iter().copied(), down);
        }
    }

    /// Read staged sample `index`.
    pub(super) fn staged_value(&self, data: &[Complex<S>], index: usize) -> S {
        if !self.is_packed() {
            data[index].re
        } else if index.is_multiple_of(2) {
            data[index / 2].re
        } else {
            data[index / 2].im
        }
    }

    /// Stage one sample in any order; real slots get a zero imaginary part.
    pub(super) fn stage_value(&self, data: &mut [Complex<S>], index: usize, value: S) {
        if self.is_packed() {
            if index.is_multiple_of(2) {
                data[index / 2].re = value;
            } else {
                data[index / 2].im = value;
            }
        } else {
            data[index] = Complex::new(value, S::ZERO);
        }
    }

    /// Measure finite `maximum` and stage range-scaled samples. Returns the
    /// restoring exponent, or `None` for an all-zero signal.
    pub(super) fn stage_scaled(
        &self,
        scratch: &mut [Complex<S>],
        maximum: S,
        stage: impl FnOnce(&mut [Complex<S>], BinaryScale<S>),
    ) -> Option<i32> {
        if maximum == S::ZERO {
            return None;
        }
        let exponent = range_exponent(maximum);
        stage(
            &mut scratch[..self.staged_len()],
            BinaryScale::new(-exponent),
        );
        Some(exponent)
    }

    /// Run the unscaled child on the staged samples. Even spectra are left
    /// unsplit; odd spectra are complete with a real DC bin, which providers
    /// must return too.
    pub(super) fn transform_staged(
        &self,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (data, work) = self.split_staging(scratch);
        match &self.algorithm {
            RealAlgorithm::Singleton => {
                crate::validation::bounded(data[0]);
            }
            RealAlgorithm::Even { complex, .. } => {
                complex.execute_bounded(&mut data[..self.n / 2], work, false, S::ONE, context)?;
            }
            RealAlgorithm::Decimated(decimation) => decimation.forward(data, work, context)?,
            RealAlgorithm::Odd { complex } => {
                complex.execute_bounded(data, work, false, S::ONE, context)?;
                data[0].im = S::ZERO;
            }
        }
        Ok(())
    }
}
/// Stage packed bins, applying `sign` to imaginary parts, and measure them.
pub(super) fn stage_bins<S: scioxide_core::RealScalar>(
    packed: &mut [Complex<S>],
    bins: impl Iterator<Item = Complex<S>>,
    sign: S,
) -> Result<S, SciOxideError> {
    complex_max(packed.iter_mut().zip(bins).map(|(slot, value)| {
        *slot = Complex::new(value.re, value.im * sign);
        value
    }))
}
