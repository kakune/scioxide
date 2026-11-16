//! Discrete wavelet transforms of finite records.
use crate::util::{boundary_index, inputs as finite};
use crate::{
    SignalBoundary, SignalRequirements, SignalWorkspace,
    util::{exact_len, inconsistent, invalid, working_sum_products},
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Argument, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar, RealScalar,
    SciOxideError,
};
mod axis;
mod tables;
mod two;
pub use two::{Wavelet2dLayout, Wavelet2dPlan, WaveletSubband};

/// Built-in wavelet filter pair.
///
/// Every pair is normalized so that the lowpass analysis taps sum to `√2`.
///
/// # References
///
/// I. Daubechies, *Ten Lectures on Wavelets*, SIAM, 1992.
///
/// A. Cohen, I. Daubechies and J.-C. Feauveau, "Biorthogonal bases of compactly
/// supported wavelets", *Comm. Pure Appl. Math.* 45(5), 485–560, 1992.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Orthonormal Haar wavelet with two taps.
    Haar,
    /// Orthonormal Daubechies wavelet with the given even filter length, 4 through 20.
    Daubechies(usize),
    /// Cohen–Daubechies–Feauveau biorthogonal spline pair of order `(i, j)`, encoded
    /// as `100·i + j`: 103, 105, 202, 204, 206, 208, 301, 303, 305, 307 or 309.
    BiorthogonalSpline(usize),
}
/// Alignment of the filters with the decimated samples, used by analysis and synthesis.
///
/// For coefficient index `j`, filter tap `k` multiplies sample `2j + k − o`,
/// where the offset `o` is zero or `⌊L/2⌋` for filter length `L`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveletPhase {
    /// Offset `o = 0`: tap `k` multiplies sample `2j + k`.
    Uncentered,
    /// Offset `o = ⌊L/2⌋`: the filter is centered on sample `2j`.
    Centered,
}
/// Boundary extension, phase and execution policy of a wavelet transform.
///
/// The default is a periodic boundary, the uncentered phase and sequential
/// execution.
///
/// ```
/// use scioxide_signal::{SignalBoundary, WaveletOptions, WaveletPhase};
/// let options = WaveletOptions::new()
///     .with_boundary(SignalBoundary::Reflect)
///     .with_phase(WaveletPhase::Centered);
/// assert_eq!(options.boundary(), SignalBoundary::Reflect);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct WaveletOptions {
    boundary: SignalBoundary,
    phase: WaveletPhase,
    execution_policy: ExecutionPolicy,
}
impl Default for WaveletOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl WaveletOptions {
    /// Returns the default options: periodic, uncentered and sequential.
    pub const fn new() -> Self {
        Self {
            boundary: SignalBoundary::Periodic,
            phase: WaveletPhase::Uncentered,
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the extension of the record beyond its ends.
    ///
    /// [`SignalBoundary::Periodic`](crate::SignalBoundary::Periodic) extends an
    /// odd-length record by repeating its last sample before it periodizes, so each
    /// band holds `⌈n/2⌉` coefficients. `Zero` and `Reflect` bands hold every
    /// position whose filter support meets the record. `Reflect` mirrors the record
    /// without repeating the endpoint samples.
    pub const fn with_boundary(mut self, boundary: SignalBoundary) -> Self {
        self.boundary = boundary;
        self
    }
    /// Sets the phase of the analysis and synthesis filters.
    pub const fn with_phase(mut self, phase: WaveletPhase) -> Self {
        self.phase = phase;
        self
    }
    /// Sets the execution policy for independent lanes of axis and 2-D transforms.
    ///
    /// At most 32 workers are used.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
    /// Returns the boundary extension.
    pub const fn boundary(self) -> SignalBoundary {
        self.boundary
    }
    /// Returns the filter phase.
    pub const fn phase(self) -> WaveletPhase {
        self.phase
    }
    /// Returns the execution policy for independent lanes.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
}
/// Analysis and synthesis filters of a built-in wavelet, in the working precision.
///
/// The filters are real. Complex samples are transformed componentwise with the
/// same filters. Tap values come from tables in `f64` and are rounded to the
/// scalar type.
///
/// ```
/// use scioxide_signal::{WaveletFamily, WaveletFilters};
/// let filters = WaveletFilters::<f64>::new(WaveletFamily::Daubechies(4))?;
/// let low = filters.analysis_low();
/// // Orthonormal lowpass taps sum to sqrt(2) with unit energy; highpass taps sum to zero.
/// assert_eq!(low.len(), 4);
/// assert!((low.iter().sum::<f64>() - 2.0_f64.sqrt()).abs() < 1e-12);
/// assert!((low.iter().map(|h| h * h).sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(filters.analysis_high().iter().sum::<f64>().abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct WaveletFilters<S: RealScalar> {
    family: WaveletFamily,
    analysis_low: Vec<S>,
    analysis_high: Vec<S>,
    synthesis_low: Vec<S>,
    synthesis_high: Vec<S>,
}
impl<S: RealScalar> WaveletFilters<S> {
    /// Returns the filters of a built-in wavelet family.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the Daubechies length or the spline code is not one of
    ///   the supported values listed on [`WaveletFamily`].
    pub fn new(family: WaveletFamily) -> Result<Self, SciOxideError> {
        let (a, b, c, d) = tables::filters(family).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("family"),
                "unsupported wavelet family or order",
            )
        })?;
        let convert = |x: &[f64]| -> Result<Vec<S>, SciOxideError> {
            let mut v = try_with_capacity(x.len(), scioxide_core::quantity!("wavelet_filters"))?;
            for &x in x {
                v.push(S::try_from_f64_lossy(x)?);
            }
            Ok(v)
        };
        Ok(Self {
            family,
            analysis_low: convert(a)?,
            analysis_high: convert(b)?,
            synthesis_low: convert(c)?,
            synthesis_high: convert(d)?,
        })
    }

    /// Returns the wavelet family.
    pub fn family(&self) -> WaveletFamily {
        self.family
    }
    /// Returns the lowpass analysis taps in increasing sample-index order.
    pub fn analysis_low(&self) -> &[S] {
        &self.analysis_low
    }
    /// Returns the highpass analysis taps in increasing sample-index order.
    pub fn analysis_high(&self) -> &[S] {
        &self.analysis_high
    }
    /// Returns the lowpass synthesis taps, applied with the same phase as analysis.
    pub fn synthesis_low(&self) -> &[S] {
        &self.synthesis_low
    }
    /// Returns the highpass synthesis taps, applied with the same phase as analysis.
    pub fn synthesis_high(&self) -> &[S] {
        &self.synthesis_high
    }
}
/// Band geometry of one decomposition level of a [`WaveletPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveletLevel {
    input_len: usize,
    band_len: usize,
    first: i128,
    detail_offset: usize,
}
impl WaveletLevel {
    /// Returns the length of the signal this level decomposes.
    pub const fn input_len(self) -> usize {
        self.input_len
    }
    /// Returns the length of the approximation band and of the detail band.
    pub const fn band_len(self) -> usize {
        self.band_len
    }
    /// Returns the coefficient index `j` of the first band entry.
    ///
    /// It is zero for a periodic boundary and can be negative for `Zero` and
    /// `Reflect`, where the band also covers filter positions that only partly
    /// overlap the record.
    pub const fn first(self) -> i128 {
        self.first
    }
    /// Returns the offset of this level's detail band in the packed coefficients.
    pub const fn detail_offset(self) -> usize {
        self.detail_offset
    }
}
/// Returns the maximum useful decomposition depth of a record of length `n`.
///
/// ```text
/// max_level = max(0, ⌊log₂(n / (L − 1))⌋)
/// ```
///
/// where `L` is the analysis filter length (biorthogonal pairs are zero-padded
/// to a common even length) and the division is integer division. Plans with
/// `Zero` or `Reflect` boundaries reject deeper transforms. Periodic plans accept
/// depths up to `⌈log₂ n⌉`.
///
/// ```
/// use scioxide_signal::{WaveletFamily, wavelet_max_level};
/// assert_eq!(wavelet_max_level(8, WaveletFamily::Daubechies(20))?, 0);
/// assert_eq!(wavelet_max_level(64, WaveletFamily::Daubechies(4))?, 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `InvalidInput` if the family order is not supported.
pub fn wavelet_max_level(n: usize, family: WaveletFamily) -> Result<usize, SciOxideError> {
    let filters = tables::filters(family).ok_or_else(|| {
        invalid(
            scioxide_core::argument!("family"),
            "unsupported wavelet family or order",
        )
    })?;
    let quotient = n / (filters.0.len() - 1);
    Ok(if quotient == 0 {
        0
    } else {
        quotient.ilog2() as usize
    })
}

fn maximum_depth(
    n: usize,
    family: WaveletFamily,
    boundary: SignalBoundary,
) -> Result<usize, SciOxideError> {
    let useful = wavelet_max_level(n, family)?;
    Ok(if boundary == SignalBoundary::Periodic {
        periodic_depth(n)
    } else {
        useful
    })
}

fn periodic_depth(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}
/// Multilevel discrete wavelet transform of a 1-D record of fixed length.
///
/// One analysis level filters the current approximation `a` (initially the
/// input `x`) with the lowpass taps `h` and the highpass taps `g`, and keeps every
/// second output:
///
/// ```text
/// a′[j] = Σₖ h[k] x̃[2j + k − o],   d[j] = Σₖ g[k] x̃[2j + k − o]
/// ```
///
/// where `x̃` is `a` extended by the boundary rule and `o` is the
/// [`WaveletPhase`] offset. The next level decomposes `a′`. Synthesis inverts the
/// transform with the synthesis filters and the same phase, and reconstructs
/// the input to rounding error.
///
/// The packed coefficients hold the final approximation first, then the detail
/// bands from the coarsest to the finest level. A transform of `n` samples with
/// filter length `L` costs `O(n L)` operations at any depth. Its accuracy follows
/// the product-sum bound in [Accuracy](crate#accuracy), propagated through the
/// levels. [`WaveletPlan::analyze_axis_into`] and
/// [`WaveletPlan::synthesize_axis_into`] transform every lane along one axis of
/// an array. One workspace serves both directions (see
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces)).
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{WaveletFamily, WaveletOptions, WaveletPlan};
/// let plan = WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 2, WaveletOptions::default())?;
/// let mut ws = plan.workspace(1)?;
/// // Two Haar levels of a ramp: analysis followed by synthesis restores it.
/// let input = [1.0, 2.0, 3.0, 4.0];
/// let mut coefficients = vec![0.0; plan.coefficient_len()];
/// let cx = ExecutionContext::sequential_native();
/// plan.analyze_into(&input, &mut coefficients, &mut ws, cx)?;
/// let restored = plan.synthesize(&coefficients)?;
/// for (a, b) in restored.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// I. Daubechies, *Ten Lectures on Wavelets*, SIAM, 1992.
///
/// S. Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed., Academic Press, 2009.
#[derive(Clone, Debug)]
pub struct WaveletPlan<T: NumericScalar> {
    filters: WaveletFilters<T::Real>,
    options: WaveletOptions,
    n: usize,
    levels: Vec<WaveletLevel>,
    coefficients: usize,
    approximation: usize,
    width: usize,
}
impl<T: NumericScalar> WaveletPlan<T> {
    /// Prepares a `levels`-level transform of records of length `n`.
    ///
    /// Zero levels is the identity transform. With `Zero` or `Reflect` boundaries,
    /// `levels` may not exceed [`wavelet_max_level`]. With a periodic boundary it
    /// may not exceed `⌈log₂ n⌉`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the family order is not supported, `levels` exceeds the
    ///   depth limit, or `n` exceeds `isize::MAX`.
    /// - `SizeOverflow` if the coefficient or workspace length overflows `usize`.
    pub fn new(
        family: WaveletFamily,
        n: usize,
        levels: usize,
        options: WaveletOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(family, n, levels, options, scioxide_core::argument!("n"))
    }
    /// Validate and lay out the filter bank.
    /// `extent` names the caller parameter that supplied `n`.
    pub(super) fn prepare(
        family: WaveletFamily,
        n: usize,
        levels: usize,
        options: WaveletOptions,
        extent: Argument,
    ) -> Result<Self, SciOxideError> {
        if levels > maximum_depth(n, family, options.boundary)? {
            return Err(invalid(
                scioxide_core::argument!("levels"),
                "levels exceed the maximum decomposition depth",
            ));
        }
        if n > isize::MAX as usize {
            return Err(invalid(
                extent,
                "record length exceeds the signed index range",
            ));
        }
        let filters = WaveletFilters::new(family)?;
        let k = filters.analysis_low.len();
        let offset = if options.phase == WaveletPhase::Centered {
            k / 2
        } else {
            0
        };
        let mut geometry = try_with_capacity(levels, scioxide_core::quantity!("wavelet_levels"))?;
        let mut current = n;
        let mut width = n;
        for _ in 0..levels {
            let (first, count) = if options.boundary == SignalBoundary::Periodic {
                (
                    0,
                    checked_add(current, 1, scioxide_core::quantity!("wavelet_padding"))? / 2,
                )
            } else {
                let first = (offset as i128 - k as i128 + 2).div_euclid(2);
                let last = (current as i128 - 1 + offset as i128).div_euclid(2);
                (
                    first,
                    usize::try_from(last - first + 1)
                        .map_err(|_| invalid(extent, "decomposition band has no samples"))?,
                )
            };
            geometry.push(WaveletLevel {
                input_len: current,
                band_len: count,
                first,
                detail_offset: 0,
            });
            current = count;
            width = width.max(checked_mul(
                count,
                2,
                scioxide_core::quantity!("wavelet_bands"),
            )?);
        }
        let approximation = current;
        for level in geometry.iter_mut().rev() {
            level.detail_offset = current;
            current = checked_add(
                current,
                level.band_len,
                scioxide_core::quantity!("wavelet_packing"),
            )?;
        }
        let plan = Self {
            filters,
            options,
            n,
            levels: geometry,
            coefficients: current,
            approximation,
            width,
        };
        // Reject unrepresentable workspace sizes during preparation.
        let _ = plan.record_requirements()?;
        Ok(plan)
    }
    /// Returns the analysis and synthesis filters.
    pub fn filters(&self) -> &WaveletFilters<T::Real> {
        &self.filters
    }
    /// Returns the record length `n`.
    pub fn input_len(&self) -> usize {
        self.n
    }
    /// Returns the number of packed coefficients.
    ///
    /// Each level adds `2·band_len − input_len` entries to `n` (see
    /// [`WaveletLevel`]). For an uncentered transform with a `Zero` or `Reflect`
    /// boundary that is `L − 2` per level, plus one for an odd input length. The
    /// centered phase can add one more pair of boundary coefficients.
    pub fn coefficient_len(&self) -> usize {
        self.coefficients
    }
    /// Returns the length of the final approximation, the first band of the packed
    /// coefficients.
    pub fn approximation_len(&self) -> usize {
        self.approximation
    }
    /// Returns the level geometry from the finest to the coarsest level.
    pub fn levels(&self) -> &[WaveletLevel] {
        &self.levels
    }
    /// Returns the workspace size for analysis and synthesis of `lanes`
    /// records at once.
    ///
    /// Use one lane for [`WaveletPlan::analyze_into`] and
    /// [`WaveletPlan::synthesize_into`], and the number of lanes along the axis
    /// for [`WaveletPlan::analyze_axis_into`] and
    /// [`WaveletPlan::synthesize_axis_into`]: the product of the extents of the
    /// other axes. Each lane holds its samples, its coefficients and the
    /// level scratch, so the size grows linearly with `lanes`, and it never
    /// falls below the size of one record.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        lanes: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(self
            .record_requirements()?
            .max(SignalRequirements::values(axis::axis_workspace(
                self, lanes,
            )?)))
    }
    /// Allocates a workspace for analysis and synthesis of `lanes` records.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, lanes: usize) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(lanes)?)
    }
    /// Workspace of one record: the staged output and two level buffers.
    fn record_requirements(&self) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(checked_add(
            self.n.max(self.coefficients),
            checked_mul(self.width, 2, scioxide_core::quantity!("wavelet_workspace"))?,
            scioxide_core::quantity!("wavelet_workspace"),
        )?))
    }

    fn offset(&self) -> i128 {
        if self.options.phase == WaveletPhase::Centered {
            (self.filters.analysis_low.len() / 2) as i128
        } else {
            0
        }
    }
    /// Paired periodic analysis shares sample loads and checks range per level.
    /// Wrapped and contiguous windows use one summation order, which the 2-D
    /// column kernel reproduces, so results do not depend on the window
    /// position or on the 2-D execution policy.
    fn analyze_periodic(&self, level: WaveletLevel, input: &[T], output: &mut [T]) -> bool {
        let n = level.input_len;
        let low = &self.filters.analysis_low;
        let high = &self.filters.analysis_high;
        if self.options.boundary != SignalBoundary::Periodic
            || !n.is_multiple_of(2)
            || level.first != 0
            || level.band_len != n / 2
        {
            return false;
        }
        let offset = self.offset() as usize % n;
        let mut start = if offset == 0 { 0 } else { n - offset };
        let (approximation, detail) = output.split_at_mut(level.band_len);
        for (a, d) in approximation.iter_mut().zip(detail.iter_mut()) {
            let (l, h) = if start + low.len() <= n {
                let x = &input[start..start + low.len()];
                periodic_row(|k| x[k], low, high)
            } else {
                periodic_row(|k| input[(start + k) % n], low, high)
            };
            *a = l;
            *d = h;
            start += 2;
            if start >= n {
                start -= n;
            }
        }
        approximation
            .iter()
            .chain(detail.iter())
            .all(|x| x.has_finite_components())
    }

    fn analyze_level(
        &self,
        level: WaveletLevel,
        input: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        if self.analyze_periodic(level, input, output) {
            return Ok(());
        }
        let n = level.input_len;
        let m = n + n % 2;
        for (q, out) in output.iter_mut().enumerate() {
            let high = q >= level.band_len;
            let j = level.first + (q % level.band_len) as i128;
            let taps = if high {
                &self.filters.analysis_high
            } else {
                &self.filters.analysis_low
            };
            let start = 2 * j - self.offset();
            if start >= 0 && start + taps.len() as i128 <= n as i128 {
                *out = crate::util::contiguous_products::<T, T::Real, false>(
                    &input[start as usize..start as usize + taps.len()],
                    taps,
                    T::from_real,
                )?;
                continue;
            }
            *out = working_sum_products(taps.iter().enumerate().map(|(k, &h)| {
                let index = 2 * j + k as i128 - self.offset();
                let sample = if self.options.boundary == SignalBoundary::Periodic {
                    input[(index.rem_euclid(m as i128) as usize).min(n - 1)]
                } else {
                    boundary_index(index, n, self.options.boundary).map_or(T::ZERO, |i| input[i])
                };
                (sample, T::from_real(h))
            }))?;
        }
        Ok(())
    }
    /// Periodic synthesis scatters each band pair into contiguous output windows.
    /// Range is checked once per level; the gather kernel recomputes exceptional ranges.
    fn synthesize_periodic(&self, level: WaveletLevel, input: &[T], output: &mut [T]) -> bool {
        let n = level.input_len;
        let low = &self.filters.synthesis_low;
        let high = &self.filters.synthesis_high;
        if self.options.boundary != SignalBoundary::Periodic
            || !n.is_multiple_of(2)
            || low.len() > n
            || level.first != 0
            || level.band_len != n / 2
        {
            return false;
        }
        output.fill(T::ZERO);
        let offset = self.offset() as usize % n;
        let mut start = if offset == 0 { 0 } else { n - offset };
        for (&a, &b) in input[..level.band_len].iter().zip(&input[level.band_len..]) {
            let count = low.len().min(n - start);
            let (head, tail) = output.split_at_mut(start);
            for ((slot, &h), &g) in tail[..count]
                .iter_mut()
                .zip(&low[..count])
                .zip(&high[..count])
            {
                *slot = *slot + a * T::from_real(h) + b * T::from_real(g);
            }
            let remaining = low.len() - count;
            for ((slot, &h), &g) in head[..remaining]
                .iter_mut()
                .zip(&low[count..])
                .zip(&high[count..])
            {
                *slot = *slot + a * T::from_real(h) + b * T::from_real(g);
            }
            start += 2;
            if start >= n {
                start -= n;
            }
        }
        output.iter().all(|x| x.has_finite_components())
    }

    fn synthesize_level(
        &self,
        level: WaveletLevel,
        input: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        if self.synthesize_periodic(level, input, output) {
            return Ok(());
        }
        let n = level.input_len;
        let m = n + n % 2;
        for (q, out) in output.iter_mut().enumerate() {
            let base = q as i128 + self.offset();
            let parity = base.rem_euclid(2) as usize;
            let low = &self.filters.synthesis_low;
            let high = &self.filters.synthesis_high;
            let last_tap = low.len() - 1 - (low.len() - 1 - parity) % 2;
            let first_band = (base - last_tap as i128).div_euclid(2) - level.first;
            let last_band = (base - parity as i128).div_euclid(2) - level.first;
            if base >= low.len() as i128 - 1
                && base < m as i128
                && first_band >= 0
                && last_band < level.band_len as i128
            {
                let bands = first_band as usize..last_band as usize + 1;
                let details = bands.start + level.band_len..bands.end + level.band_len;
                *out = working_sum_products(
                    input[bands]
                        .iter()
                        .rev()
                        .zip(input[details].iter().rev())
                        .zip(
                            low[parity..]
                                .iter()
                                .step_by(2)
                                .zip(high[parity..].iter().step_by(2)),
                        )
                        .flat_map(|((&a, &b), (&h, &g))| {
                            [(a, T::from_real(h)), (b, T::from_real(g))]
                        }),
                )?;
                continue;
            }
            *out = working_sum_products(
                self.filters
                    .synthesis_low
                    .iter()
                    .zip(&self.filters.synthesis_high)
                    .enumerate()
                    .skip(((q as i128 + self.offset()).rem_euclid(2)) as usize)
                    .step_by(2)
                    .flat_map(|(k, (&h, &g))| {
                        let index = q as i128 + self.offset() - k as i128;
                        let index = if self.options.boundary == SignalBoundary::Periodic {
                            index.rem_euclid(m as i128)
                        } else {
                            index
                        };
                        let j = index.div_euclid(2) - level.first;
                        let valid =
                            index.rem_euclid(2) == 0 && j >= 0 && j < level.band_len as i128;
                        let (a, b) = if valid {
                            (input[j as usize], input[level.band_len + j as usize])
                        } else {
                            (T::ZERO, T::ZERO)
                        };
                        [(a, T::from_real(h)), (b, T::from_real(g))]
                    }),
            )?;
        }
        Ok(())
    }
    fn raw(
        &self,
        input: &[T],
        output: &mut [T],
        work: &mut [T],
        inverse: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.levels.is_empty() {
            output.copy_from_slice(input);
            return Ok(());
        }
        if let [level] = self.levels.as_slice() {
            context.check()?;
            return if inverse {
                self.synthesize_level(*level, input, output)
            } else {
                self.analyze_level(*level, input, output)
            };
        }
        let (a, b) = work.split_at_mut(self.width);
        if inverse {
            a[..self.approximation].copy_from_slice(&input[..self.approximation]);
            for &level in self.levels.iter().rev() {
                context.check()?;
                a[level.band_len..2 * level.band_len].copy_from_slice(
                    &input[level.detail_offset..level.detail_offset + level.band_len],
                );
                self.synthesize_level(level, &a[..2 * level.band_len], &mut b[..level.input_len])?;
                a[..level.input_len].copy_from_slice(&b[..level.input_len]);
            }
            output.copy_from_slice(&a[..self.n]);
        } else {
            a[..self.n].copy_from_slice(input);
            for &level in &self.levels {
                context.check()?;
                self.analyze_level(level, &a[..level.input_len], &mut b[..2 * level.band_len])?;
                output[level.detail_offset..level.detail_offset + level.band_len]
                    .copy_from_slice(&b[level.band_len..2 * level.band_len]);
                a[..level.band_len].copy_from_slice(&b[..level.band_len]);
            }
            output[..self.approximation].copy_from_slice(&a[..self.approximation]);
        }
        Ok(())
    }
    /// Validates lengths and finiteness for one direction, then stages the
    /// transform in the workspace and copies it to the output after success.
    fn transform(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        inverse: bool,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, m) = if inverse {
            (self.coefficients, self.n)
        } else {
            (self.n, self.coefficients)
        };
        exact_len(n, input.len())?;
        exact_len(m, output.len())?;
        let buffers = ws.buffers(&self.record_requirements()?)?;
        finite(input)?;
        cx.check()?;
        let (stage, scratch) = buffers.values.split_at_mut(self.n.max(self.coefficients));
        self.raw(input, &mut stage[..m], scratch, inverse, cx)?;
        cx.check()?;
        output.copy_from_slice(&stage[..m]);
        Ok(())
    }

    /// Computes the packed wavelet coefficients of `input`.
    ///
    /// `output` holds [`WaveletPlan::coefficient_len`] values. The execution
    /// context is checked for cancellation between levels.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have length `n`, `output` does
    ///   not have length [`WaveletPlan::coefficient_len`], or the workspace is too
    ///   small.
    /// - `NonFinite` if an input sample is NaN or infinite.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn analyze_into(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform(input, output, ws, false, cx)
    }
    /// Reconstructs a record of length `n` from packed coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`WaveletPlan::analyze_into`], with the roles of the two
    /// lengths exchanged.
    pub fn synthesize_into(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform(input, output, ws, true, cx)
    }
    /// Computes the packed wavelet coefficients of every lane of `input` along
    /// axis `axis` and writes them to `output`.
    ///
    /// Each lane of `n` samples is transformed as [`WaveletPlan::analyze_into`]
    /// transforms a record. `output` has the shape of `input` with `n`
    /// replaced by [`WaveletPlan::coefficient_len`], and both views may be
    /// strided. The lanes run in parallel under the execution policy of the
    /// plan's [`WaveletOptions`]. `ws` needs the size of
    /// [`WaveletPlan::workspace_requirements`] for the number of lanes; with it
    /// the call does not allocate. The coefficients are staged in `ws` and
    /// written to `output` after every lane succeeds. The cost is that of
    /// [`WaveletPlan::analyze_into`] per lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of
    ///   `input`.
    /// - `DimensionMismatch` if the axis length of `input` is not `n`, `output`
    ///   does not have the expected shape, or `ws` is too short.
    /// - The errors of [`WaveletPlan::analyze_into`] for each lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{WaveletFamily, WaveletOptions, WaveletPlan};
    /// let plan = WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 1, WaveletOptions::default())?;
    /// // Two records of four samples, one per row; each is constant in pairs.
    /// let x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    /// let layout = StridedLayout::new(&[2, 4], &[4, 1])?;
    /// let mut ws = plan.workspace(2)?;
    /// let mut coefficients = [0.0; 8];
    /// let output = NdViewMut::new(&mut coefficients, layout)?;
    /// let cx = ExecutionContext::sequential_native();
    /// plan.analyze_axis_into(NdView::new(&x, layout)?, 1, output, &mut ws, cx)?;
    /// // The Haar details of each row vanish; its approximations are √2 times the pairs.
    /// assert!(coefficients[2].abs() < 1e-12 && coefficients[7].abs() < 1e-12);
    /// assert!((coefficients[4] - 3.0 * 2.0_f64.sqrt()).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn analyze_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform_axis(input, axis, output, ws, false, cx)
    }
    /// Reconstructs every lane of `output` along axis `axis` from the packed
    /// coefficients in `input`.
    ///
    /// Each lane of [`WaveletPlan::coefficient_len`] coefficients is
    /// reconstructed as [`WaveletPlan::synthesize_into`] reconstructs a record.
    /// `output` has the shape of `input` with that length replaced by `n`.
    /// Lanes, workspace, staging and cost are those of
    /// [`WaveletPlan::analyze_axis_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`WaveletPlan::analyze_axis_into`], with the sample and
    /// coefficient lengths exchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{WaveletFamily, WaveletOptions, WaveletPlan};
    /// let plan = WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 2, WaveletOptions::default())?;
    /// // The coefficients of two ramps, one record per row.
    /// let x = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, 0.0];
    /// let coefficients = [plan.analyze(&x[..4])?, plan.analyze(&x[4..])?].concat();
    /// let layout = StridedLayout::new(&[2, 4], &[4, 1])?;
    /// let mut ws = plan.workspace(2)?;
    /// let mut y = [0.0; 8];
    /// let input = NdView::new(&coefficients, layout)?;
    /// let cx = ExecutionContext::sequential_native();
    /// plan.synthesize_axis_into(input, 1, NdViewMut::new(&mut y, layout)?, &mut ws, cx)?;
    /// assert!(y.iter().zip(x).all(|(a, b)| (a - b).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn synthesize_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform_axis(input, axis, output, ws, true, cx)
    }
    /// Validates the view shapes for one direction, borrows the lane blocks of
    /// every lane and runs the staged axis transform.
    fn transform_axis(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<T>,
        inverse: bool,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, m) = if inverse {
            (self.coefficients, self.n)
        } else {
            (self.n, self.coefficients)
        };
        let shape = input.layout().shape().extents();
        let lanes = crate::axis::Geometry::new(shape, axis, n, m)?.batches;
        crate::util::check_axis_shape(shape, axis, m, output.layout().shape().extents())?;
        let buffers = ws.buffers(&SignalRequirements::values(axis::axis_workspace(
            self, lanes,
        )?))?;
        axis::axis_transform(self, input, output, axis, buffers.values, inverse, cx)
    }
    /// Computes and returns the packed wavelet coefficients of `input`.
    ///
    /// # Errors
    ///
    /// The errors of [`WaveletPlan::analyze_into`], and `AllocationFailure` if the
    /// result or workspace cannot be allocated.
    pub fn analyze(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        self.allocate(input, false)
    }
    /// Reconstructs and returns a record of length `n` from packed coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`WaveletPlan::synthesize_into`], and `AllocationFailure` if
    /// the result or workspace cannot be allocated.
    pub fn synthesize(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        self.allocate(input, true)
    }
    fn allocate(&self, input: &[T], inverse: bool) -> Result<Vec<T>, SciOxideError> {
        let mut out = try_filled(
            if inverse { self.n } else { self.coefficients },
            T::ZERO,
            scioxide_core::quantity!("wavelet_output"),
        )?;
        let mut ws = SignalWorkspace::reserved(&self.record_requirements()?)?;
        self.transform(
            input,
            &mut out,
            &mut ws,
            inverse,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
}
/// One periodic analysis row of samples `x(k)`: two- and four-tap filters sum
/// `(x0*h0 + x1*h1) + (x2*h2 + x3*h3)`, longer filters fold in tap order.
/// The 2-D column kernel spells out the same two- and four-tap expressions.
#[inline(always)]
fn periodic_row<T: NumericScalar>(
    x: impl Fn(usize) -> T,
    low: &[T::Real],
    high: &[T::Real],
) -> (T, T) {
    let product = |i: usize, taps: &[T::Real]| x(i) * T::from_real(taps[i]);
    match low.len() {
        2 => (
            product(0, low) + product(1, low),
            product(0, high) + product(1, high),
        ),
        4 => (
            (product(0, low) + product(1, low)) + (product(2, low) + product(3, low)),
            (product(0, high) + product(1, high)) + (product(2, high) + product(3, high)),
        ),
        _ => (0..low.len()).fold((T::ZERO, T::ZERO), |(l, h), k| {
            (l + product(k, low), h + product(k, high))
        }),
    }
}
#[cfg(test)]
mod tests;
