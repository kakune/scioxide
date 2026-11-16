//! Peak detection by ridge lines of a Ricker continuous wavelet transform.
use crate::util::{
    capacity, check_real_slice_finite, empty, finite, invalid, scalar_from_usize_rounded as scalar,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled, try_reserve, try_with_capacity};
use scioxide_core::{Capacity, ExecutionContext, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};
/// Kernel support and ridge-linking criteria of a [`RidgePlan`].
///
/// The default uses a kernel half-support of 10 samples, links maxima at most
/// one sample apart on consecutive scales with no missing scales, keeps ridges
/// of any length and accepts every positive coefficient. There is no
/// signal-to-noise criterion: use the minimum strength and minimum length
/// instead. [`RidgePlan::new`] validates the options against the scales.
///
/// ```
/// use scioxide_signal::RidgeOptions;
/// let options = RidgeOptions::<f64>::default()
///     .with_min_length(2)
///     .with_max_gap(1);
/// assert_eq!((options.support_radius(), options.max_distance()), (10, 1));
/// assert_eq!((options.min_length(), options.max_gap()), (2, 1));
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct RidgeOptions<S: RealScalar> {
    support_radius: usize,
    max_distance: usize,
    max_gap: usize,
    min_length: usize,
    min_strength: S,
}
impl<S: RealScalar> Default for RidgeOptions<S> {
    fn default() -> Self {
        Self {
            support_radius: 10,
            max_distance: 1,
            max_gap: 0,
            min_length: 1,
            min_strength: S::ZERO,
        }
    }
}
impl<S: RealScalar> RidgeOptions<S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the kernel half-support `R` in samples, the same for every scale.
    pub fn with_support_radius(mut self, support_radius: usize) -> Self {
        self.support_radius = support_radius;
        self
    }
    /// Sets the largest index shift, in samples, between linked maxima.
    pub fn with_max_distance(mut self, max_distance: usize) -> Self {
        self.max_distance = max_distance;
        self
    }
    /// Sets the largest number of consecutive scales a ridge may skip.
    pub fn with_max_gap(mut self, max_gap: usize) -> Self {
        self.max_gap = max_gap;
        self
    }
    /// Sets the minimum number of linked maxima of a reported ridge.
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }
    /// Sets the smallest transform coefficient a maximum may have to be linked.
    pub fn with_min_strength(mut self, min_strength: S) -> Self {
        self.min_strength = min_strength;
        self
    }
    /// Returns the kernel half-support `R` in samples.
    pub fn support_radius(self) -> usize {
        self.support_radius
    }
    /// Returns the largest index shift between linked maxima.
    pub fn max_distance(self) -> usize {
        self.max_distance
    }
    /// Returns the largest number of consecutive scales a ridge may skip.
    pub fn max_gap(self) -> usize {
        self.max_gap
    }
    /// Returns the minimum number of linked maxima of a reported ridge.
    pub fn min_length(self) -> usize {
        self.min_length
    }
    /// Returns the smallest coefficient a maximum may have to be linked.
    pub fn min_strength(self) -> S {
        self.min_strength
    }
}
/// Peak reported by [`RidgePlan::find_peaks_into`]: the finest maximum of a ridge.
///
/// `Default` returns an all-zero record for initializing output buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgePeak<S: RealScalar> {
    index: usize,
    scale_index: usize,
    length: usize,
    strength: S,
}
impl<S: RealScalar> Default for RidgePeak<S> {
    fn default() -> Self {
        Self {
            index: 0,
            scale_index: 0,
            length: 0,
            strength: S::ZERO,
        }
    }
}
impl<S: RealScalar> RidgePeak<S> {
    /// Returns the sample index of the ridge at its finest linked scale.
    pub fn index(&self) -> usize {
        self.index
    }
    /// Returns the index of that scale in [`RidgePlan::scales`].
    pub fn scale_index(&self) -> usize {
        self.scale_index
    }
    /// Returns the number of linked maxima.
    pub fn length(&self) -> usize {
        self.length
    }
    /// Returns the transform coefficient at the finest linked scale.
    pub fn strength(&self) -> S {
        self.strength
    }
}
/// One linked ridge during detection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RidgeState<S: RealScalar> {
    peak: RidgePeak<S>,
    last_step: usize,
}
impl<S: RealScalar> Default for RidgeState<S> {
    fn default() -> Self {
        Self {
            peak: RidgePeak::default(),
            last_step: 0,
        }
    }
}
/// Storage that [`RidgePlan::find_peaks_into`] needs for inputs up to a maximum length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct RidgeRequirements {
    coefficients: usize,
    maxima: usize,
    ridges: usize,
}
impl RidgeRequirements {
    /// Returns the number of ridge records, `⌊n/2⌋ · K` for maximum input length
    /// `n` and `K` scales.
    pub const fn ridges_len(&self) -> usize {
        self.ridges
    }
}
/// Workspace of [`RidgePlan::find_peaks_into`].
///
/// Create it with [`RidgePlan::workspace`] for the longest input. Each call
/// reuses the storage and does not grow it.
#[derive(Clone, Debug)]
pub struct RidgeWorkspace<S: RealScalar> {
    coefficients: Vec<S>,
    maxima: Vec<RidgePeak<S>>,
    ridges: Vec<RidgeState<S>>,
    linkable: Vec<(usize, usize)>,
}
impl<S: RealScalar> Default for RidgeWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> RidgeWorkspace<S> {
    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            maxima: Vec::new(),
            ridges: Vec::new(),
            linkable: Vec::new(),
        }
    }
    /// Grows the workspace to meet `requirements`, keeping any larger storage.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(&mut self, requirements: &RidgeRequirements) -> Result<(), SciOxideError> {
        grow(
            &mut self.coefficients,
            requirements.coefficients,
            S::ZERO,
            scioxide_core::quantity!("ridge_coefficients"),
        )?;
        grow(
            &mut self.maxima,
            requirements.maxima,
            RidgePeak::default(),
            scioxide_core::quantity!("ridge_maxima"),
        )?;
        grow(
            &mut self.ridges,
            requirements.ridges,
            RidgeState::default(),
            scioxide_core::quantity!("ridge_workspace"),
        )?;
        grow(
            &mut self.linkable,
            requirements.ridges,
            (0, 0),
            scioxide_core::quantity!("ridge_linkable"),
        )
    }
    /// Reports whether the workspace meets `requirements`, without growing it.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub fn check(&self, requirements: &RidgeRequirements) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.coefficients.len() >= requirements.coefficients
                && self.maxima.len() >= requirements.maxima
                && self.ridges.len() >= requirements.ridges
                && self.linkable.len() >= requirements.ridges
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
}
fn grow<E: Clone>(
    values: &mut Vec<E>,
    len: usize,
    zero: E,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if values.len() < len {
        try_reserve(values, len - values.len(), quantity)?;
        values.resize(len, zero);
    }
    Ok(())
}
/// Peak detector that links maxima of a Ricker wavelet transform across scales.
///
/// For each scale `a`, the transform coefficient of sample `i` is
///
/// ```text
/// c_a[i] = Σ_{|k| ≤ R} ψ_a(k) x[i + k],
/// ψ_a(t) = 2 / (√(3a) π^(1/4)) · (1 − (t/a)²) · exp(−t²/(2a²))
/// ```
///
/// where `R` is the support radius, the Ricker (Mexican hat) wavelet `ψ_a` has
/// unit `L²` norm, and samples beyond the ends repeat the nearest end sample.
/// Positive local maxima of `c_a` at interior samples (plateau midpoints, as in
/// [`find_peaks_into`](crate::find_peaks_into)) are linked into ridges from the
/// coarsest to the finest scale. Stronger maxima link first. Each joins the
/// nearest ridge within the maximum distance that has not yet been extended on
/// this scale and has skipped at most the allowed number of scales. Ties go to
/// the stronger ridge, then the lower index. A maximum that joins no ridge
/// starts one.
///
/// The plan stores the `K(2R + 1)` kernel values of its `K` scales.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{RidgeOptions, RidgePeak, RidgePlan};
/// let options = RidgeOptions::default()
///     .with_support_radius(4)
///     .with_max_distance(2);
/// let plan = RidgePlan::new(&[1.0_f64, 2.0], options)?;
/// let mut workspace = plan.workspace(5)?;
/// let mut peaks = [RidgePeak::default(); 2];
/// let input = [0.0, 0.0, 1.0, 0.0, 0.0];
/// let cx = ExecutionContext::sequential_native();
/// let count = plan.find_peaks_into(&input, &mut peaks, &mut workspace, cx)?;
/// // A unit impulse gives positive maxima on both scales, so a ridge survives.
/// assert!(count > 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. Du, W. A. Kibbe and S. M. Lin, "Improved peak detection in mass spectrum by
/// incorporating continuous wavelet transform-based pattern matching",
/// *Bioinformatics* 22(17), 2059–2065, 2006.
#[derive(Clone, Debug)]
pub struct RidgePlan<S: RealScalar> {
    scales: Vec<S>,
    options: RidgeOptions<S>,
    radius: isize,
    support: usize,
    kernels: Vec<Scaled<S>>,
}
impl<S: RealScalar> RidgePlan<S> {
    /// Prepares the kernels for positive, strictly increasing scales in samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `scales` is empty, not positive or not strictly
    ///   increasing, the support radius is zero or exceeds `isize::MAX`, the
    ///   minimum length is zero or exceeds `scales.len()`, or the minimum strength
    ///   is negative.
    /// - `NonFinite` if a scale or the minimum strength is NaN or infinite.
    /// - `NumericalInstability` if a kernel value overflows.
    /// - `SizeOverflow` if the kernel storage overflows `usize`.
    /// - `AllocationFailure` if the kernels cannot be stored.
    pub fn new(scales: &[S], options: RidgeOptions<S>) -> Result<Self, SciOxideError> {
        check_real_slice_finite(scales)?;
        check_real_slice_finite(&[options.min_strength])?;
        if scales.is_empty() {
            return Err(empty(
                scioxide_core::argument!("scales"),
                "scales must be nonempty",
            ));
        }
        if scales.iter().any(|&s| s <= S::ZERO) || scales.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(
                scioxide_core::argument!("scales"),
                "scales must be positive and strictly increasing",
            ));
        }
        if options.support_radius == 0 {
            return Err(invalid(
                scioxide_core::argument!("options.support_radius"),
                "support radius must be positive",
            ));
        }
        if options.min_length == 0 || options.min_length > scales.len() {
            return Err(invalid(
                scioxide_core::argument!("options.min_length"),
                "minimum ridge length must lie in 1..=scales.len()",
            ));
        }
        if options.min_strength < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.min_strength"),
                "minimum strength must be nonnegative",
            ));
        }
        let radius = isize::try_from(options.support_radius).map_err(|_| {
            invalid(
                scioxide_core::argument!("options.support_radius"),
                "support radius exceeds the signed index range",
            )
        })?;
        let support = options
            .support_radius
            .checked_mul(2)
            .and_then(|r| r.checked_add(1))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ridge_support"))
            })?;
        let mut kernels = try_with_capacity(
            checked_mul(
                scales.len(),
                support,
                scioxide_core::quantity!("ridge_kernels"),
            )?,
            scioxide_core::quantity!("ridge_kernels"),
        )?;
        let cutoff = scalar::<S>(64)?;
        for &scale in scales {
            let normalization =
                finite(S::TWO / (scalar::<S>(3)?.sqrt() * scale.sqrt() * S::PI.sqrt().sqrt()))?;
            for offset in -radius..=radius {
                let u = scalar::<S>(offset.unsigned_abs())? / scale;
                let value = if !u.is_finite() || u > cutoff {
                    S::ZERO
                } else {
                    finite(normalization * (S::ONE - u * u) * (-S::HALF * u * u).exp())?
                };
                kernels.push(Scaled::from_value(value)?);
            }
        }
        Ok(Self {
            scales: crate::util::copy(scales)?,
            options,
            radius,
            support,
            kernels,
        })
    }
    /// Returns the scales in increasing order.
    pub fn scales(&self) -> &[S] {
        &self.scales
    }
    /// Returns the options.
    pub fn options(&self) -> RidgeOptions<S> {
        self.options
    }
    /// Returns the workspace size for inputs of at most `max_input_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the number of ridge records overflows `usize`.
    pub fn workspace_requirements(
        &self,
        max_input_len: usize,
    ) -> Result<RidgeRequirements, SciOxideError> {
        Ok(RidgeRequirements {
            coefficients: max_input_len,
            maxima: max_input_len / 2,
            ridges: checked_mul(
                max_input_len / 2,
                self.scales.len(),
                scioxide_core::quantity!("ridge_workspace"),
            )?,
        })
    }
    /// Allocates a workspace for inputs of at most `max_input_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the number of ridge records overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, max_input_len: usize) -> Result<RidgeWorkspace<S>, SciOxideError> {
        let mut workspace = RidgeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(max_input_len)?)?;
        Ok(workspace)
    }
    /// Finds the peaks of `input` as ridges of its Ricker wavelet transform and
    /// returns them sorted by sample index.
    ///
    /// The result has one [`RidgePeak`] per ridge with at least the minimum
    /// length, as described in [`find_peaks_into`](Self::find_peaks_into), whose
    /// cost it shares. This allocates a workspace for `input.len()` samples and
    /// `input.len()` records, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a transform coefficient overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the workspace or the result
    ///   cannot be sized or allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{RidgeOptions, RidgePlan};
    /// let options = RidgeOptions::default().with_support_radius(8);
    /// let plan = RidgePlan::new(&[1.0_f64, 2.0, 3.0], options)?;
    /// let input: Vec<f64> = (0..41)
    ///     .map(|k| (-0.1 * (k as f64 - 20.0).powi(2)).exp())
    ///     .collect();
    /// let peaks = plan.find_peaks(&input)?;
    /// // The bump at sample 20 is a ridge across all three scales.
    /// assert_eq!(peaks.len(), 1);
    /// assert_eq!((peaks[0].index(), peaks[0].length()), (20, 3));
    /// assert!(peaks[0].strength() > 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn find_peaks(&self, input: &[S]) -> Result<Vec<RidgePeak<S>>, SciOxideError> {
        let mut workspace = self.workspace(input.len())?;
        let mut output = try_filled(
            input.len(),
            RidgePeak::default(),
            scioxide_core::quantity!("ridge_output"),
        )?;
        let count = self.find_peaks_into(
            input,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        output.truncate(count);
        Ok(output)
    }
    /// Finds the peaks of `input` as ridges of its Ricker wavelet transform,
    /// writing them into `output`.
    ///
    /// Writes one [`RidgePeak`] per ridge with at least the minimum length to the
    /// front of `output`, sorted by sample index, and returns their number. When
    /// several ridges end at the same sample, the one with the finest scale, then
    /// the longest, is reported. The workspace must meet
    /// [`RidgePlan::workspace_requirements`] for `input.len()`.
    ///
    /// The transform costs `O(n K R)` for `n` samples. Linking costs
    /// `O(m log m)` per scale for `m` maxima and open ridges.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the workspace is too small, or `output` is shorter
    ///   than the number of peaks.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a transform coefficient overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn find_peaks_into(
        &self,
        input: &[S],
        output: &mut [RidgePeak<S>],
        workspace: &mut RidgeWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<usize, SciOxideError> {
        check_real_slice_finite(input)?;
        let requirements = self.workspace_requirements(input.len())?;
        capacity(requirements.coefficients, workspace.coefficients.len())?;
        capacity(requirements.maxima, workspace.maxima.len())?;
        capacity(requirements.ridges, workspace.ridges.len())?;
        capacity(requirements.ridges, workspace.linkable.len())?;
        context.check()?;
        let coefficients = &mut workspace.coefficients[..requirements.coefficients];
        let maxima = &mut workspace.maxima[..requirements.maxima];
        let ridges = &mut workspace.ridges[..requirements.ridges];
        // `(index, slot)` of every ridge eligible at the current step, sorted.
        let linkable = &mut workspace.linkable[..requirements.ridges];
        let options = self.options;
        let mut count = 0;
        let mut eligible = 0;
        for (step, scale_index) in (0..self.scales.len()).rev().enumerate() {
            let found = self.maxima_into(input, scale_index, coefficients, maxima)?;
            let created = count;
            for &peak in &maxima[..found] {
                // The nearest, then strongest, then lowest-index ridge not yet
                // linked at this step; equal ridges prefer the earliest created.
                let low = peak.index.saturating_sub(options.max_distance);
                let high = peak.index.saturating_add(options.max_distance);
                let from = linkable[..eligible].partition_point(|&(index, _)| index < low);
                let mut best: Option<usize> = None;
                for &(index, slot) in &linkable[from..eligible] {
                    if index > high {
                        break;
                    }
                    if ridges[slot].last_step == step {
                        continue;
                    }
                    let candidate = ridges[slot].peak;
                    let better = best.is_none_or(|b| {
                        let incumbent = ridges[b].peak;
                        candidate
                            .index
                            .abs_diff(peak.index)
                            .cmp(&incumbent.index.abs_diff(peak.index))
                            .then_with(|| {
                                incumbent
                                    .strength
                                    .partial_cmp(&candidate.strength)
                                    .unwrap_or(core::cmp::Ordering::Equal)
                            })
                            .then(candidate.index.cmp(&incumbent.index))
                            .is_lt()
                    });
                    if better {
                        best = Some(slot);
                    }
                }
                if let Some(i) = best {
                    let length = ridges[i].peak.length + 1;
                    ridges[i] = RidgeState {
                        peak: RidgePeak { length, ..peak },
                        last_step: step,
                    };
                } else {
                    ridges[count] = RidgeState {
                        peak,
                        last_step: step,
                    };
                    count += 1;
                }
            }
            // Keep ridges that can still link at the next step, add this
            // step's new ridges, and re-sort by their current indices.
            let mut kept = 0;
            for k in 0..eligible {
                let slot = linkable[k].1;
                if ridges[slot].last_step.saturating_add(options.max_gap) >= step {
                    linkable[kept] = (ridges[slot].peak.index, slot);
                    kept += 1;
                }
            }
            for (slot, ridge) in ridges.iter().enumerate().take(count).skip(created) {
                linkable[kept] = (ridge.peak.index, slot);
                kept += 1;
            }
            linkable[..kept].sort_unstable();
            eligible = kept;
        }
        let mut retained = 0;
        // Sorted by index, then finest scale, then longest.
        for i in 0..count {
            if ridges[i].peak.length >= options.min_length {
                ridges[retained] = ridges[i];
                retained += 1;
            }
        }
        ridges[..retained].sort_unstable_by(|a, b| {
            a.peak
                .index
                .cmp(&b.peak.index)
                .then(a.peak.scale_index.cmp(&b.peak.scale_index))
                .then(b.peak.length.cmp(&a.peak.length))
        });
        // Several terminated ridges may share a final sample. Keep the finest, then longest.
        let mut unique = 0;
        for i in 0..retained {
            if unique == 0 || ridges[i].peak.index != ridges[unique - 1].peak.index {
                ridges[unique] = ridges[i];
                unique += 1;
            }
        }
        capacity(unique, output.len())?;
        context.check()?;
        output
            .iter_mut()
            .zip(&ridges[..unique])
            .for_each(|(y, r)| *y = r.peak);
        Ok(unique)
    }
    /// Ricker coefficients of `scale_index` and their positive interior
    /// maxima, ordered by decreasing strength then index: stronger maxima
    /// claim a ridge first.
    fn maxima_into(
        &self,
        input: &[S],
        scale_index: usize,
        coefficients: &mut [S],
        maxima: &mut [RidgePeak<S>],
    ) -> Result<usize, SciOxideError> {
        let kernel = &self.kernels[scale_index * self.support..(scale_index + 1) * self.support];
        for (i, value) in coefficients[..input.len()].iter_mut().enumerate() {
            let mut sum = ScaledAccumulator::ZERO;
            for (offset, &coefficient) in (-self.radius..=self.radius).zip(kernel) {
                let j = crate::util::clamped_index(i, offset, input.len());
                sum.add_product(Scaled::from_value(input[j])?, coefficient)?;
            }
            *value = sum.round()?;
        }
        let mut found = 0;
        let mut i = 1;
        while i + 1 < input.len() {
            let start = i;
            while i + 1 < input.len() && coefficients[i + 1] == coefficients[start] {
                i += 1;
            }
            if i + 1 < input.len()
                && coefficients[start] > coefficients[start - 1]
                && coefficients[i] > coefficients[i + 1]
                && coefficients[start] > S::ZERO
                && coefficients[start] >= self.options.min_strength
            {
                maxima[found] = RidgePeak {
                    index: start + (i - start) / 2,
                    scale_index,
                    length: 1,
                    strength: coefficients[start],
                };
                found += 1;
            }
            i += 1;
        }
        maxima[..found].sort_unstable_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.index.cmp(&b.index))
        });
        Ok(found)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Independent linker: every maximum examines every ridge ever created.
    fn reference(plan: &RidgePlan<f64>, input: &[f64]) -> Vec<RidgePeak<f64>> {
        let options = plan.options;
        let mut coefficients = vec![0.0; input.len()];
        let mut maxima = vec![RidgePeak::default(); input.len() / 2];
        let mut ridges: Vec<RidgeState<f64>> = Vec::new();
        for (step, scale_index) in (0..plan.scales.len()).rev().enumerate() {
            let found = plan
                .maxima_into(input, scale_index, &mut coefficients, &mut maxima)
                .unwrap();
            for &peak in &maxima[..found] {
                let best = ridges
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.last_step < step
                            && step - r.last_step - 1 <= options.max_gap
                            && r.peak.index.abs_diff(peak.index) <= options.max_distance
                    })
                    .min_by(|(_, a), (_, b)| {
                        a.peak
                            .index
                            .abs_diff(peak.index)
                            .cmp(&b.peak.index.abs_diff(peak.index))
                            .then(b.peak.strength.total_cmp(&a.peak.strength))
                            .then(a.peak.index.cmp(&b.peak.index))
                    })
                    .map(|(i, _)| i);
                match best {
                    Some(i) => {
                        let length = ridges[i].peak.length + 1;
                        ridges[i] = RidgeState {
                            peak: RidgePeak { length, ..peak },
                            last_step: step,
                        };
                    }
                    None => ridges.push(RidgeState {
                        peak,
                        last_step: step,
                    }),
                }
            }
        }
        let mut kept: Vec<_> = ridges
            .into_iter()
            .filter(|r| r.peak.length >= options.min_length)
            .map(|r| r.peak)
            .collect();
        kept.sort_by(|a, b| {
            a.index
                .cmp(&b.index)
                .then(a.scale_index.cmp(&b.scale_index))
                .then(b.length.cmp(&a.length))
        });
        kept.dedup_by_key(|p| p.index);
        kept
    }

    #[test]
    fn indexed_linking_matches_the_all_ridges_scan() {
        let mut state = 0x2545_f491_u32;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state) / f64::from(u32::MAX) - 0.5
        };
        let scales: Vec<f64> = (1..=6).map(f64::from).collect();
        for n in [0, 1, 3, 17, 200, 1500] {
            let noise: Vec<f64> = (0..n).map(|_| uniform()).collect();
            // Quantized samples create equal-strength maxima and tied ridges.
            let ties: Vec<f64> = noise.iter().map(|x| (x * 4.0).round()).collect();
            for input in [&noise, &ties] {
                for (max_distance, max_gap, min_length) in [
                    (1, 0, 1),
                    (0, 0, 1),
                    (2, 1, 2),
                    (5, 3, 3),
                    (usize::MAX, usize::MAX, 1),
                ] {
                    let options = RidgeOptions::default()
                        .with_support_radius(6)
                        .with_max_distance(max_distance)
                        .with_max_gap(max_gap)
                        .with_min_length(min_length);
                    let plan = RidgePlan::new(&scales, options).unwrap();
                    let mut workspace = plan.workspace(input.len()).unwrap();
                    let mut output = vec![RidgePeak::default(); input.len() * scales.len()];
                    let count = plan
                        .find_peaks_into(
                            input,
                            &mut output,
                            &mut workspace,
                            ExecutionContext::sequential_native(),
                        )
                        .unwrap();
                    assert_eq!(
                        output[..count],
                        reference(&plan, input),
                        "n={n} distance={max_distance} gap={max_gap}"
                    );
                }
            }
        }
    }
}
