//! Scratch storage for scalar descriptive statistics, trimming, transforms and
//! spacing entropy.
use crate::information::DifferentialEntropyOptions;
use crate::{
    InformationValue, KernelWeights, RawOrderWorkspace, RobustStatistic, SigmaClipOptions,
    SigmaClipResult, TrimSide,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{RealScalar, Scaled, SciOxideError, SolveContext};

/// Reusable scratch for scalar descriptive statistics, trimming and real
/// transforms. Only [`Self::reserve`] allocates.
pub(crate) struct RawDescriptiveWorkspace<S: RealScalar> {
    pub(crate) work: Vec<S>,
    pub(crate) stage: Vec<S>,
    pub(crate) order: RawOrderWorkspace<S>,
    deviations: Vec<Scaled<f64>>,
}
impl<S: RealScalar> Default for RawDescriptiveWorkspace<S> {
    fn default() -> Self {
        Self {
            work: Vec::new(),
            stage: Vec::new(),
            order: RawOrderWorkspace::default(),
            deviations: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawDescriptiveWorkspace<S> {
    pub(crate) fn required_bytes(n: usize) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[
            (n, core::mem::size_of::<S>() * 2),
            (n, core::mem::size_of::<Scaled<f64>>()),
            (1, RawOrderWorkspace::<S>::required_bytes(n)?),
        ])
    }
    /// Grows the scratch to hold `samples` observations. On error the
    /// existing storage is kept.
    /// # Errors
    /// Returns size-overflow and allocation errors.
    pub(crate) fn reserve(&mut self, samples: usize) -> Result<(), SciOxideError> {
        if self.capacity() >= samples {
            return Ok(());
        }
        let work = try_filled(
            samples,
            S::ZERO,
            scioxide_core::quantity!("descriptive_work"),
        )?;
        let stage = try_filled(
            samples,
            S::ZERO,
            scioxide_core::quantity!("descriptive_stage"),
        )?;
        let mut order = RawOrderWorkspace::default();
        order.reserve(samples)?;
        let deviations = try_filled(
            samples,
            Scaled::<f64>::ZERO,
            scioxide_core::quantity!("descriptive_deviations"),
        )?;
        *self = Self {
            work,
            stage,
            order,
            deviations,
        };
        Ok(())
    }
    /// Number of observations the scratch holds.
    pub(crate) fn capacity(&self) -> usize {
        self.work.len()
    }

    /// Evaluates one statistic. Means, sums, variances, standard deviations and
    /// extrema need no reserved scratch.
    /// # Errors
    /// Rejects short scratch, invalid or empty samples, unsupported weights,
    /// undefined statistics and results that overflow.
    pub(crate) fn evaluate(
        &mut self,
        values: &[S],
        weights: Option<KernelWeights<'_, S>>,
        statistic: RobustStatistic<S>,
    ) -> Result<S, SciOxideError> {
        crate::robust::evaluate(
            values,
            weights,
            statistic,
            &mut self.work,
            &mut self.deviations,
        )
    }

    /// Writes the sorted observations kept by trimming to the front of `output`
    /// and returns their number.
    /// # Errors
    /// Rejects invalid proportions, empty results, non-finite samples and
    /// short storage.
    pub(crate) fn trim_into(
        &mut self,
        values: &[S],
        proportion: S,
        side: TrimSide,
        output: &mut [S],
    ) -> Result<usize, SciOxideError> {
        crate::robust::trim_into(values, proportion, side, output, &mut self.work)
    }

    /// Geometric z-score `(log x − mean(log x)) / sd(log x)`.
    /// # Errors
    /// Rejects nonpositive or non-finite samples, a zero or undefined scale,
    /// `n ≤ ddof`, short scratch and a mismatched output length.
    pub(crate) fn geometric_zscore_into(
        &mut self,
        values: &[S],
        ddof: usize,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::standardize::geometric_zscore_into(
            values,
            ddof,
            output,
            &mut self.work,
            &mut self.stage,
        )
    }

    /// O'Brien's transform of one group with `n ≥ 3`. The mean of the
    /// transformed values equals the group's unbiased sample variance.
    /// # Errors
    /// Rejects invalid samples, short storage and results that overflow.
    pub(crate) fn obrien_transform_into(
        &mut self,
        values: &[S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::standardize::obrien_transform_into(values, output, &mut self.stage)
    }

    /// Iterative sigma clipping; writes the kept observations and their input
    /// indices to the front of `output` and `indices`.
    /// # Errors
    /// Rejects invalid options, non-finite data, short buffers, clipping that
    /// removes every observation, numerical failures and execution failures.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sigma_clip_into(
        &mut self,
        values: &[S],
        low: S,
        high: S,
        options: SigmaClipOptions<S>,
        context: &mut SolveContext<'_, '_>,
        output: &mut [S],
        indices: &mut [usize],
    ) -> Result<SigmaClipResult<S>, SciOxideError> {
        crate::standardize::sigma_clip_into(
            values,
            low,
            high,
            options,
            context,
            output,
            indices,
            &mut self.stage,
        )
    }

    /// Spacing estimate of differential entropy. Sorting costs O(n log n);
    /// the Correa estimator then costs O(n·m) and the others O(n).
    /// # Errors
    /// Rejects invalid or empty samples, short scratch, an invalid window or
    /// base, and finite results that overflow. A zero spacing gives −∞.
    pub(crate) fn differential_entropy_with_options(
        &mut self,
        values: &[S],
        options: DifferentialEntropyOptions<S>,
    ) -> Result<InformationValue<S>, SciOxideError> {
        crate::information::differential_entropy_sorted(values, options, &mut self.work)
    }
}
