//! Peak finding and peak properties of real sequences.
use crate::util::{
    capacity, check_real_slice_finite, finite, inconsistent, invalid,
    scalar_from_usize_rounded as scalar,
};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_reserve};
use scioxide_core::{Argument, Capacity, ExecutionContext, Interval, RealScalar, SciOxideError};

/// Local maximum of a real sequence and its prominence and width, in sample units.
///
/// A peak is a plateau `x[s] = … = x[e]` with `0 < s ≤ e < n − 1` whose
/// neighbors `x[s − 1]` and `x[e + 1]` are both lower. Its bases are the
/// positions of the minimum of `x` between the plateau and the nearest sample on
/// each side that is higher than the peak (or the end of the record). Among equal
/// minima, the one nearest the peak is the base. With peak height `h`, the prominence is
///
/// ```text
/// P = h − max(x[left_base], x[right_base])
/// ```
///
/// The width is measured on the contour `h − r·P` for relative height `r`. On
/// each side, a scan outward from the plateau stops at the first sample at or
/// below the contour, or at the base. The crossing is interpolated linearly
/// between that sample and its inner neighbor.
///
/// `Default` returns an all-zero record for initializing output buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak<S: RealScalar> {
    index: usize,
    plateau_start: usize,
    plateau_end: usize,
    left_base: usize,
    right_base: usize,
    prominence: S,
    left_crossing: S,
    right_crossing: S,
    width: S,
}
impl<S: RealScalar> Default for Peak<S> {
    fn default() -> Self {
        Self {
            index: 0,
            plateau_start: 0,
            plateau_end: 0,
            left_base: 0,
            right_base: 0,
            prominence: S::ZERO,
            left_crossing: S::ZERO,
            right_crossing: S::ZERO,
            width: S::ZERO,
        }
    }
}
impl<S: RealScalar> Peak<S> {
    /// Returns the plateau midpoint `s + ⌊(e − s)/2⌋`.
    pub fn index(&self) -> usize {
        self.index
    }
    /// Returns the first index `s` of the plateau.
    pub fn plateau_start(&self) -> usize {
        self.plateau_start
    }
    /// Returns the last index `e` of the plateau.
    pub fn plateau_end(&self) -> usize {
        self.plateau_end
    }
    /// Returns the left base: the minimum position left of the plateau before a higher sample.
    pub fn left_base(&self) -> usize {
        self.left_base
    }
    /// Returns the right base: the minimum position right of the plateau before a higher sample.
    pub fn right_base(&self) -> usize {
        self.right_base
    }
    /// Returns the prominence `P`, the peak height minus the higher of the two base heights.
    pub fn prominence(&self) -> S {
        self.prominence
    }
    /// Returns the interpolated position where the contour meets the left flank.
    pub fn left_crossing(&self) -> S {
        self.left_crossing
    }
    /// Returns the interpolated position where the contour meets the right flank.
    pub fn right_crossing(&self) -> S {
        self.right_crossing
    }
    /// Returns the width, `right_crossing − left_crossing`.
    ///
    /// At relative height zero the contour is the peak height, so the width of a
    /// plateau is `plateau_end − plateau_start`.
    pub fn width(&self) -> S {
        self.width
    }
}
/// Selection criteria for [`find_peaks_with_options`] and [`find_peaks_into`].
///
/// The interval criteria are inclusive and apply to every peak first. Distance
/// selection then runs on the peaks that pass them. The default accepts every
/// peak, has no minimum distance and measures widths at half prominence.
/// [`find_peaks_with_options`] and [`find_peaks_into`] validate the options.
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_signal::PeakOptions;
/// let options = PeakOptions::default()
///     .with_prominence(Interval::new(0.5_f64, 10.0)?)
///     .with_distance(3);
/// assert_eq!(options.distance(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PeakOptions<S: RealScalar> {
    height: Option<Interval<S>>,
    prominence: Option<Interval<S>>,
    width: Option<Interval<S>>,
    plateau_size: Option<(usize, usize)>,
    distance: usize,
    relative_height: S,
}
impl<S: RealScalar> Default for PeakOptions<S> {
    fn default() -> Self {
        Self {
            height: None,
            prominence: None,
            width: None,
            plateau_size: None,
            distance: 0,
            relative_height: S::HALF,
        }
    }
}
impl<S: RealScalar> PeakOptions<S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Accepts only peaks whose height lies in `height`.
    pub fn with_height(mut self, height: Interval<S>) -> Self {
        self.height = Some(height);
        self
    }
    /// Accepts only peaks whose prominence lies in `prominence`.
    pub fn with_prominence(mut self, prominence: Interval<S>) -> Self {
        self.prominence = Some(prominence);
        self
    }
    /// Accepts only peaks whose width, in samples, lies in `width`.
    pub fn with_width(mut self, width: Interval<S>) -> Self {
        self.width = Some(width);
        self
    }
    /// Accepts only peaks whose plateau has between `low` and `high` samples,
    /// inclusive, where `plateau_size = (low, high)` and `0 < low ≤ high`.
    pub fn with_plateau_size(mut self, plateau_size: (usize, usize)) -> Self {
        self.plateau_size = Some(plateau_size);
        self
    }
    /// Sets the minimum distance `d` between selected peaks.
    ///
    /// Peaks are taken in decreasing order of height, lower index first among
    /// equal heights. Each taken peak removes the remaining peaks whose midpoints
    /// lie less than `d` samples away. Zero and one select every peak.
    pub fn with_distance(mut self, distance: usize) -> Self {
        self.distance = distance;
        self
    }
    /// Sets the relative height `r` in `[0, 1]` of the width contour `h − r·P`.
    ///
    /// Zero measures at the peak height, one at the higher base.
    pub fn with_relative_height(mut self, relative_height: S) -> Self {
        self.relative_height = relative_height;
        self
    }
    /// Returns the height criterion, if any.
    pub fn height(self) -> Option<Interval<S>> {
        self.height
    }
    /// Returns the prominence criterion, if any.
    pub fn prominence(self) -> Option<Interval<S>> {
        self.prominence
    }
    /// Returns the width criterion, if any.
    pub fn width(self) -> Option<Interval<S>> {
        self.width
    }
    /// Returns the inclusive plateau-size limits, if any.
    pub fn plateau_size(self) -> Option<(usize, usize)> {
        self.plateau_size
    }
    /// Returns the minimum distance between selected peaks.
    pub fn distance(self) -> usize {
        self.distance
    }
    /// Returns the relative height of the width contour, one half by default.
    pub fn relative_height(self) -> S {
        self.relative_height
    }
}
/// Storage that [`find_peaks_into`] needs for inputs up to a maximum length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct PeakRequirements {
    records: usize,
    tree: usize,
}
impl PeakRequirements {
    /// Returns the number of peak records, `⌊n/2⌋` for maximum input length `n`.
    pub const fn records_len(&self) -> usize {
        self.records
    }
    /// Returns the length of each of the two segment trees: twice `n` rounded up
    /// to a power of two.
    pub const fn tree_len(&self) -> usize {
        self.tree
    }
}
/// Workspace of [`find_peaks_into`].
///
/// Allocate it once with [`PeakWorkspace::workspace`], or grow it with
/// [`PeakWorkspace::try_reserve`], for the longest input. Each call reuses the
/// storage and does not grow it.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{Peak, PeakOptions, PeakWorkspace, find_peaks_into};
/// let input = [0.0_f64, 1.0, 0.0];
/// let mut workspace = PeakWorkspace::workspace(input.len())?;
/// let mut peaks = [Peak::default(); 1];
/// let options = PeakOptions::default();
/// let cx = ExecutionContext::sequential_native();
/// let count = find_peaks_into(&input, options, &mut peaks, &mut workspace, cx)?;
/// // The only local maximum of [0, 1, 0] is sample 1.
/// assert_eq!(count, 1);
/// assert_eq!(peaks[0].index(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct PeakWorkspace<S: RealScalar> {
    candidates: Vec<Peak<S>>,
    priority: Vec<usize>,
    suppressed: Vec<bool>,
    minimum: Vec<S>,
    maximum: Vec<S>,
}
impl<S: RealScalar> Default for PeakWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> PeakWorkspace<S> {
    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            candidates: Vec::new(),
            priority: Vec::new(),
            suppressed: Vec::new(),
            minimum: Vec::new(),
            maximum: Vec::new(),
        }
    }
    /// Returns the storage [`find_peaks_into`] needs for inputs of at most
    /// `max_input_len` samples.
    ///
    /// A tree length that does not fit in `usize` saturates, so reserving it
    /// fails with `AllocationFailure`.
    pub const fn workspace_requirements(max_input_len: usize) -> PeakRequirements {
        PeakRequirements {
            records: max_input_len / 2,
            tree: match max_input_len.checked_next_power_of_two() {
                Some(leaves) => leaves.saturating_mul(2),
                None => usize::MAX,
            },
        }
    }
    /// Allocates a workspace for inputs of at most `max_input_len` samples.
    ///
    /// This is [`new`](Self::new) followed by [`try_reserve`](Self::try_reserve)
    /// with [`workspace_requirements(max_input_len)`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(max_input_len: usize) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(max_input_len))?;
        Ok(workspace)
    }
    /// Grows the workspace to meet `requirements`, keeping any larger storage.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(&mut self, requirements: &PeakRequirements) -> Result<(), SciOxideError> {
        grow(
            &mut self.candidates,
            requirements.records,
            Peak::default(),
            scioxide_core::quantity!("peak_candidates"),
        )?;
        grow(
            &mut self.priority,
            requirements.records,
            0,
            scioxide_core::quantity!("peak_priority"),
        )?;
        grow(
            &mut self.suppressed,
            requirements.records,
            false,
            scioxide_core::quantity!("peak_suppression"),
        )?;
        grow(
            &mut self.minimum,
            requirements.tree,
            S::ZERO,
            scioxide_core::quantity!("peak_minimum_tree"),
        )?;
        grow(
            &mut self.maximum,
            requirements.tree,
            S::ZERO,
            scioxide_core::quantity!("peak_maximum_tree"),
        )
    }
    /// Reports whether the workspace meets `requirements`, without growing it.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub fn check(&self, requirements: &PeakRequirements) -> Result<Capacity, SciOxideError> {
        Ok(
            if [
                self.candidates.len(),
                self.priority.len(),
                self.suppressed.len(),
            ]
            .into_iter()
            .all(|len| len >= requirements.records)
                && self.minimum.len() >= requirements.tree
                && self.maximum.len() >= requirements.tree
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
/// The four nearest-sample searches of peak geometry over a finite record.
///
/// [`Scan`] answers them with the defining outward scans; [`Trees`] runs the
/// same scans for at most [`NEAR`] samples and answers longer searches from
/// extremum trees, so both produce identical records.
trait Search<S: RealScalar> {
    /// Nearest minimum at or left of `start`, scanning left until a sample
    /// above `height` or the record start.
    fn left_base(&mut self, start: usize, height: S) -> usize;
    /// Nearest minimum at or right of `end`, scanning right until a sample
    /// above `height` or the record end.
    fn right_base(&mut self, end: usize, height: S) -> usize;
    /// Largest `i` in `left..=start` with `x[i] <= contour`, else `left`.
    fn left_crossing(&mut self, left: usize, start: usize, contour: S) -> usize;
    /// Smallest `i` in `end..=right` with `x[i] <= contour`, else `right`.
    fn right_crossing(&mut self, end: usize, right: usize, contour: S) -> usize;
}
/// Left base scan over `floor..start`; `None` when it stops at `floor > 0`
/// before meeting a higher sample.
fn left_base_scan<S: RealScalar>(x: &[S], start: usize, height: S, floor: usize) -> Option<usize> {
    let mut left = start;
    for i in (floor..start).rev() {
        if x[i] > height {
            return Some(left);
        }
        if x[i] < x[left] {
            left = i;
        }
    }
    (floor == 0).then_some(left)
}
/// Right base scan over `end + 1..ceiling`; `None` when it stops at
/// `ceiling < x.len()` before meeting a higher sample.
fn right_base_scan<S: RealScalar>(x: &[S], end: usize, height: S, ceiling: usize) -> Option<usize> {
    let mut right = end;
    for (i, &v) in x.iter().enumerate().take(ceiling).skip(end + 1) {
        if v > height {
            return Some(right);
        }
        if v < x[right] {
            right = i;
        }
    }
    (ceiling == x.len()).then_some(right)
}
/// Left crossing scan down to `floor >= left`; `None` when it stops above
/// the contour before reaching `left`.
fn left_crossing_scan<S: RealScalar>(
    x: &[S],
    left: usize,
    start: usize,
    contour: S,
    floor: usize,
) -> Option<usize> {
    let mut i = start;
    while i > floor && x[i] > contour {
        i -= 1;
    }
    (i == left || x[i] <= contour).then_some(i)
}
/// Right crossing scan up to `ceiling <= right`; `None` when it stops above
/// the contour before reaching `right`.
fn right_crossing_scan<S: RealScalar>(
    x: &[S],
    end: usize,
    right: usize,
    contour: S,
    ceiling: usize,
) -> Option<usize> {
    let mut i = end;
    while i < ceiling && x[i] > contour {
        i += 1;
    }
    (i == right || x[i] <= contour).then_some(i)
}
/// Unbounded outward scans: `O(n)` per search, no storage.
struct Scan<'a, S>(&'a [S]);
impl<S: RealScalar> Search<S> for Scan<'_, S> {
    fn left_base(&mut self, start: usize, height: S) -> usize {
        left_base_scan(self.0, start, height, 0).unwrap_or(start)
    }
    fn right_base(&mut self, end: usize, height: S) -> usize {
        right_base_scan(self.0, end, height, self.0.len()).unwrap_or(end)
    }
    fn left_crossing(&mut self, left: usize, start: usize, contour: S) -> usize {
        left_crossing_scan(self.0, left, start, contour, left).unwrap_or(left)
    }
    fn right_crossing(&mut self, end: usize, right: usize, contour: S) -> usize {
        right_crossing_scan(self.0, end, right, contour, right).unwrap_or(right)
    }
}
/// Samples a [`Trees`] search scans before querying the trees. Below a few
/// hundred samples a linear scan is cheaper than four tree descents, so ordinary
/// records keep their scan cost while adversarial records stay linear.
const NEAR: usize = 512;
/// Implicit minimum and maximum segment trees over `leaves` (a power of two)
/// padded with infinities, built on the first search longer than [`NEAR`]
/// samples: `O(n)` once, then `O(log n)` per search.
struct Trees<'a, S> {
    x: &'a [S],
    minimum: &'a mut [S],
    maximum: &'a mut [S],
    leaves: usize,
    built: bool,
}
impl<'a, S: RealScalar> Trees<'a, S> {
    /// Trees over `x` in borrowed prefixes `2 * leaves` long.
    fn new(x: &'a [S], minimum: &'a mut [S], maximum: &'a mut [S]) -> Self {
        let leaves = minimum.len() / 2;
        Self {
            x,
            minimum,
            maximum,
            leaves,
            built: false,
        }
    }
    fn build(&mut self) {
        if self.built {
            return;
        }
        let (minimum, maximum, leaves) = (&mut *self.minimum, &mut *self.maximum, self.leaves);
        for (i, (lo, hi)) in minimum[leaves..]
            .iter_mut()
            .zip(&mut maximum[leaves..])
            .enumerate()
        {
            (*lo, *hi) = self
                .x
                .get(i)
                .map_or((S::INFINITY, S::NEG_INFINITY), |&v| (v, v));
        }
        for node in (1..leaves).rev() {
            let (a, b) = (minimum[2 * node], minimum[2 * node + 1]);
            minimum[node] = if b < a { b } else { a };
            let (a, b) = (maximum[2 * node], maximum[2 * node + 1]);
            maximum[node] = if b > a { b } else { a };
        }
        self.built = true;
    }
    /// Nearest `i` in `lo..=hi` (the last with `last`) with `x[i] > level`.
    fn above(&mut self, lo: usize, hi: usize, level: S, last: bool) -> Option<usize> {
        self.build();
        Descent {
            tree: &*self.maximum,
            lo,
            hi,
            last,
            prune: |m: S| m <= level,
        }
        .find(1, 0, self.leaves)
    }
    /// Nearest `i` in `lo..=hi` (the last with `last`) with `x[i] <= level`.
    fn at_most(&mut self, lo: usize, hi: usize, level: S, last: bool) -> Option<usize> {
        self.build();
        Descent {
            tree: &*self.minimum,
            lo,
            hi,
            last,
            prune: |m: S| m > level,
        }
        .find(1, 0, self.leaves)
    }
    /// Minimum of the nonempty range `lo..=hi`.
    fn least(&mut self, lo: usize, hi: usize) -> S {
        self.build();
        let (mut lo, mut hi) = (lo + self.leaves, hi + self.leaves + 1);
        let mut value = S::INFINITY;
        while lo < hi {
            if lo % 2 == 1 {
                if self.minimum[lo] < value {
                    value = self.minimum[lo];
                }
                lo += 1;
            }
            if hi % 2 == 1 {
                hi -= 1;
                if self.minimum[hi] < value {
                    value = self.minimum[hi];
                }
            }
            lo /= 2;
            hi /= 2;
        }
        value
    }
}
/// One nearest-leaf search: the first (or, with `last`, final) leaf in
/// `lo..=hi` whose subtree extremum `prune` never rejects.
struct Descent<'a, S, P> {
    tree: &'a [S],
    lo: usize,
    hi: usize,
    last: bool,
    prune: P,
}
impl<S: RealScalar, P: Fn(S) -> bool> Descent<'_, S, P> {
    /// Search below `node`, which spans leaves `start..start + width`.
    fn find(&self, node: usize, start: usize, width: usize) -> Option<usize> {
        if start > self.hi || start + width <= self.lo || (self.prune)(self.tree[node]) {
            return None;
        }
        if width == 1 {
            return Some(start);
        }
        let half = width / 2;
        let (near, far) = if self.last {
            ((2 * node + 1, start + half), (2 * node, start))
        } else {
            ((2 * node, start), (2 * node + 1, start + half))
        };
        self.find(near.0, near.1, half)
            .or_else(|| self.find(far.0, far.1, half))
    }
}
impl<S: RealScalar> Search<S> for Trees<'_, S> {
    fn left_base(&mut self, start: usize, height: S) -> usize {
        if let Some(left) = left_base_scan(self.x, start, height, start.saturating_sub(NEAR)) {
            return left;
        }
        // The rightmost minimum after the nearest higher sample.
        let low = self.above(0, start - 1, height, true).map_or(0, |i| i + 1);
        let floor = self.least(low, start);
        self.at_most(low, start, floor, true).unwrap_or(start)
    }
    fn right_base(&mut self, end: usize, height: S) -> usize {
        let ceiling = end.saturating_add(NEAR + 1).min(self.x.len());
        if let Some(right) = right_base_scan(self.x, end, height, ceiling) {
            return right;
        }
        // The leftmost minimum before the nearest higher sample.
        let last = self.x.len() - 1;
        let high = self
            .above(end + 1, last, height, false)
            .map_or(last, |i| i - 1);
        let floor = self.least(end, high);
        self.at_most(end, high, floor, false).unwrap_or(end)
    }
    fn left_crossing(&mut self, left: usize, start: usize, contour: S) -> usize {
        let floor = start.saturating_sub(NEAR).max(left);
        left_crossing_scan(self.x, left, start, contour, floor)
            .or_else(|| self.at_most(left, start, contour, true))
            .unwrap_or(left)
    }
    fn right_crossing(&mut self, end: usize, right: usize, contour: S) -> usize {
        let ceiling = end.saturating_add(NEAR).min(right);
        right_crossing_scan(self.x, end, right, contour, ceiling)
            .or_else(|| self.at_most(end, right, contour, false))
            .unwrap_or(right)
    }
}
/// Geometry of the peak at `index`; an index that is not an interior maximum
/// is reported as `index_argument`.
fn geometry<S: RealScalar>(
    x: &[S],
    search: &mut impl Search<S>,
    index: usize,
    rel: S,
    index_argument: Argument,
) -> Result<Peak<S>, SciOxideError> {
    if index == 0 || index >= x.len().saturating_sub(1) {
        return Err(invalid(index_argument, "peak index must be interior"));
    }
    let height = x[index];
    let mut start = index;
    let mut end = index;
    while start > 0 && x[start - 1] == height {
        start -= 1;
    }
    while end + 1 < x.len() && x[end + 1] == height {
        end += 1;
    }
    if start == 0 || end + 1 == x.len() || x[start - 1] >= height || x[end + 1] >= height {
        return Err(inconsistent(
            index_argument,
            "index must identify an interior maximum",
        ));
    }
    let left = search.left_base(start, height);
    let right = search.right_base(end, height);
    let prominence = finite(height - x[left].max(x[right]))?;
    let contour = finite(height - rel * prominence)?;
    let li = search.left_crossing(left, start, contour);
    let ri = search.right_crossing(end, right, contour);
    let mut lc = scalar(li)?;
    let mut rc = scalar(ri)?;
    if x[li] < contour {
        lc += crossing_fraction(contour, x[li], x[li + 1])?;
    }
    if x[ri] < contour {
        rc -= crossing_fraction(contour, x[ri], x[ri - 1])?;
    }
    Ok(Peak {
        index: start + (end - start) / 2,
        plateau_start: start,
        plateau_end: end,
        left_base: left,
        right_base: right,
        prominence,
        left_crossing: lc,
        right_crossing: rc,
        width: finite(rc - lc)?,
    })
}
/// Shared input checks; `relative_height_argument` names the caller's
/// relative-height parameter.
fn validate<S: RealScalar>(
    x: &[S],
    relative_height: S,
    relative_height_argument: Argument,
) -> Result<(), SciOxideError> {
    check_real_slice_finite(x)?;
    check_real_slice_finite(&[relative_height])?;
    if !relative_height.is_finite() || relative_height < S::ZERO || relative_height > S::ONE {
        return Err(invalid(
            relative_height_argument,
            "relative height must lie in [0, 1]",
        ));
    }
    Ok(())
}
/// Computes the prominence and width of given peaks.
///
/// Each entry of `indices` may be any index on the plateau of a peak (see
/// [`Peak`]). Entry `i` of the result is the record of `indices[i]`, equal to
/// the record [`find_peaks_with_options`] reports for that peak with widths
/// measured at `relative_height`. Each peak costs `O(n)` in the worst case for
/// `n` samples. This allocates `indices.len()` records, calls
/// [`peak_properties_into`] and runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` if an index is not interior or does not lie on a peak
///   plateau, or `relative_height` is outside `[0, 1]`.
/// - `NonFinite` if a sample or `relative_height` is NaN or infinite.
/// - `NumericalInstability` if a prominence or width overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// let input = [0.0_f64, 2.0, 4.0, 2.0, 0.0];
/// let peaks = scioxide_signal::peak_properties(&input, &[2], 0.5)?;
/// // Prominence 4 − 0 = 4; the half-prominence contour (height 2) meets samples 1 and 3.
/// assert!((peaks[0].prominence() - 4.0).abs() < 1e-12);
/// assert!((peaks[0].width() - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn peak_properties<S: RealScalar>(
    input: &[S],
    indices: &[usize],
    relative_height: S,
) -> Result<Vec<Peak<S>>, SciOxideError> {
    let mut output = try_filled(
        indices.len(),
        Peak::default(),
        scioxide_core::quantity!("peak_output"),
    )?;
    peak_properties_into(
        input,
        indices,
        relative_height,
        &mut output,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes the prominence and width of given peaks into `output`.
///
/// Each entry of `indices` may be any index on the plateau of a peak (see
/// [`Peak`]). `output[i]` receives the record of `indices[i]`, equal to the
/// record [`find_peaks_into`] reports for that peak, with widths measured at
/// `relative_height`. No workspace is needed. Each peak costs `O(n)` in the
/// worst case, and every peak is measured twice: once to validate all indices
/// and once to write the output.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{Peak, peak_properties_into};
/// let input = [0.0_f64, 2.0, 4.0, 2.0, 0.0];
/// let mut peaks = [Peak::default(); 1];
/// let cx = ExecutionContext::sequential_native();
/// peak_properties_into(&input, &[2], 0.5, &mut peaks, cx)?;
/// // Prominence 4 − 0 = 4; the half-prominence contour (height 2) meets samples 1 and 3.
/// assert_eq!((peaks[0].prominence(), peaks[0].width()), (4.0, 2.0));
/// assert_eq!((peaks[0].left_base(), peaks[0].right_base()), (0, 4));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `InvalidInput` if an index is not interior or does not lie on a peak
///   plateau, or `relative_height` is outside `[0, 1]`.
/// - `DimensionMismatch` if `output` is shorter than `indices`.
/// - `NonFinite` if a sample or `relative_height` is NaN or infinite.
/// - `NumericalInstability` if a prominence or width overflows.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
pub fn peak_properties_into<S: RealScalar>(
    input: &[S],
    indices: &[usize],
    relative_height: S,
    output: &mut [Peak<S>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    validate(
        input,
        relative_height,
        scioxide_core::argument!("relative_height"),
    )?;
    capacity(indices.len(), output.len())?;
    context.check()?;
    let measure = |&i: &usize| {
        geometry(
            input,
            &mut Scan(input),
            i,
            relative_height,
            scioxide_core::argument!("indices"),
        )
    };
    indices.iter().try_for_each(|i| measure(i).map(|_| ()))?;
    context.check()?;
    for (slot, i) in output.iter_mut().zip(indices) {
        *slot = measure(i)?;
    }
    Ok(())
}
/// Finds every peak of a real sequence with its prominence and width.
///
/// A peak is a plateau higher than both of its neighbors (see [`Peak`]), and
/// widths are measured at half prominence. This is [`find_peaks_with_options`]
/// with [`PeakOptions::default`], which accepts every peak and sets no minimum
/// distance. The geometry of `p` local maxima of `n` samples costs
/// `O(n + p log n)`.
///
/// # Errors
///
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` if a prominence or width overflows.
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// let peaks = scioxide_signal::find_peaks(&[0.0_f64, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0])?;
/// let indices: Vec<_> = peaks.iter().map(|peak| peak.index()).collect();
/// assert_eq!(indices, [1, 3, 5]);
/// // Sample 3 rises 3 above the minima on both sides.
/// assert!((peaks[1].prominence() - 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn find_peaks<S: RealScalar>(input: &[S]) -> Result<Vec<Peak<S>>, SciOxideError> {
    find_peaks_with_options(input, PeakOptions::default())
}
/// Finds the peaks of a real sequence that meet the criteria in `options`.
///
/// Returns the selected peaks (see [`Peak`]) in increasing index order. The
/// interval criteria of [`PeakOptions`] apply to every peak first, and distance
/// selection then keeps the highest peaks that lie at least
/// `options.distance()` samples apart. This allocates a [`PeakWorkspace`] and
/// `⌊n/2⌋` peak records for `n` samples, calls [`find_peaks_into`], whose cost
/// it shares, and runs with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `options.relative_height()` is outside
///   `[0, 1]`, or the plateau-size limits do not satisfy `0 < low ≤ high`.
/// - `NonFinite` if a sample or the relative height is NaN or infinite.
/// - `NumericalInstability` if a prominence or width overflows.
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_signal::{PeakOptions, find_peaks_with_options};
/// let input = [0.0_f64, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
/// // The peak at sample 1 has prominence one and is filtered out.
/// let options = PeakOptions::new().with_prominence(Interval::new(1.5, 10.0)?);
/// let peaks = find_peaks_with_options(&input, options)?;
/// let indices: Vec<_> = peaks.iter().map(|peak| peak.index()).collect();
/// assert_eq!(indices, [3, 5]);
/// assert!((peaks[1].prominence() - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn find_peaks_with_options<S: RealScalar>(
    input: &[S],
    options: PeakOptions<S>,
) -> Result<Vec<Peak<S>>, SciOxideError> {
    let requirements = PeakWorkspace::<S>::workspace_requirements(input.len());
    let mut workspace = PeakWorkspace::workspace(input.len())?;
    let mut output = try_filled(
        requirements.records_len(),
        Peak::default(),
        scioxide_core::quantity!("peak_output"),
    )?;
    let count = find_peaks_into(
        input,
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    output.truncate(count);
    Ok(output)
}
/// Finds the peaks of a real sequence that meet the criteria in `options`,
/// writing them into `output`.
///
/// Writes the selected peaks (see [`Peak`]) to the front of `output` in
/// increasing index order and returns their number. `output` needs room only for
/// the selected peaks. The workspace must meet
/// [`PeakWorkspace::workspace_requirements`] for `input.len()`.
///
/// Bases and contour crossings are found by scanning at most 512 samples
/// outward and then querying minimum and maximum segment trees, so the
/// geometry of `p` local maxima costs `O(n + p log n)`. Distance selection costs
/// `O(p log p)`.
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_signal::{Peak, PeakOptions, PeakWorkspace, find_peaks_into};
/// let input = [0.0_f64, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
/// let mut workspace = PeakWorkspace::workspace(input.len())?;
/// // The peak at sample 1 has prominence one and is filtered out.
/// let options = PeakOptions::default().with_prominence(Interval::new(1.5, 10.0)?);
/// let mut peaks = [Peak::default(); 3];
/// let cx = ExecutionContext::sequential_native();
/// let count = find_peaks_into(&input, options, &mut peaks, &mut workspace, cx)?;
/// let indices: Vec<_> = peaks[..count].iter().map(|peak| peak.index()).collect();
/// assert_eq!(indices, [3, 5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `InvalidInput` if `options.relative_height()` is outside `[0, 1]`, or the
///   plateau-size limits do not satisfy `0 < low ≤ high`.
/// - `DimensionMismatch` if the workspace is too small, or `output` is shorter
///   than the number of selected peaks.
/// - `NonFinite` if a sample or the relative height is NaN or infinite.
/// - `NumericalInstability` if a prominence or width overflows.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
pub fn find_peaks_into<S: RealScalar>(
    input: &[S],
    options: PeakOptions<S>,
    output: &mut [Peak<S>],
    workspace: &mut PeakWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    validate(
        input,
        options.relative_height,
        scioxide_core::argument!("options.relative_height"),
    )?;
    if let Some((lo, hi)) = options.plateau_size
        && (lo == 0 || lo > hi)
    {
        return Err(invalid(
            scioxide_core::argument!("options.plateau_size"),
            "plateau size bounds must satisfy 0 < low <= high",
        ));
    }
    let requirements = PeakWorkspace::<S>::workspace_requirements(input.len());
    if workspace.check(&requirements)? != Capacity::Sufficient {
        let available = workspace
            .candidates
            .len()
            .min(workspace.priority.len())
            .min(workspace.suppressed.len());
        capacity(requirements.records, available)?;
        capacity(
            requirements.tree,
            workspace.minimum.len().min(workspace.maximum.len()),
        )?;
    }
    context.check()?;
    let scratch = &mut workspace.candidates;
    let mut search = Trees::new(
        input,
        &mut workspace.minimum[..requirements.tree],
        &mut workspace.maximum[..requirements.tree],
    );
    let mut count = 0;
    let mut i = 1;
    while i + 1 < input.len() {
        let start = i;
        while i + 1 < input.len() && input[i + 1] == input[start] {
            i += 1;
        }
        if i + 1 < input.len() && input[start] > input[start - 1] && input[i] > input[i + 1] {
            let p = geometry(
                input,
                &mut search,
                start,
                options.relative_height,
                scioxide_core::argument!("input"),
            )?;
            let accept = options.height.is_none_or(|v| v.contains(input[start]))
                && options.prominence.is_none_or(|v| v.contains(p.prominence))
                && options.width.is_none_or(|v| v.contains(p.width))
                && options
                    .plateau_size
                    .is_none_or(|(lo, hi)| (lo..=hi).contains(&(i - start + 1)));
            if accept {
                scratch[count] = p;
                count += 1;
            }
        }
        i += 1;
    }
    let candidates = &mut scratch[..count];
    let mut selected = count;
    if options.distance > 1 {
        let priority = &mut workspace.priority[..count];
        let suppressed = &mut workspace.suppressed[..count];
        suppressed.fill(false);
        for (i, slot) in priority.iter_mut().enumerate() {
            *slot = i;
        }
        priority.sort_unstable_by(|&a, &b| {
            input[candidates[b].index]
                .partial_cmp(&input[candidates[a].index])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &i in priority.iter() {
            if suppressed[i] {
                continue;
            }
            let index = candidates[i].index;
            let mut left = i;
            while left > 0 && index - candidates[left - 1].index < options.distance {
                left -= 1;
                suppressed[left] = true;
            }
            let mut right = i + 1;
            while right < count && candidates[right].index - index < options.distance {
                suppressed[right] = true;
                right += 1;
            }
        }
        // Selected centers are separated by distance, so any candidate lies in
        // at most two suppression neighborhoods. The scans above are O(p).
        selected = 0;
        for (i, &skip) in suppressed.iter().enumerate() {
            if !skip {
                candidates[selected] = candidates[i];
                selected += 1;
            }
        }
    }
    capacity(selected, output.len())?;
    context.check()?;
    output[..selected].copy_from_slice(&candidates[..selected]);
    Ok(selected)
}
/// Fractional contour crossing, with overflow.
fn crossing_fraction<S: RealScalar>(contour: S, low: S, high: S) -> Result<S, SciOxideError> {
    let scale = low.abs().max(high.abs());
    finite((contour / scale - low / scale) / (high / scale - low / scale))
}
