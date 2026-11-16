//! Lane traversal of one selected axis of a dense N-D candidate.
//!
//! A lane is the `len` values at `start + i * stride` along one axis. Lanes are
//! enumerated in row-major order of the remaining extents and grouped into
//! runs whose starts advance by one constant step, so each run's first start is
//! decoded once. Contiguous lanes are handed to kernels in place; strided lanes
//! are gathered a block at a time into lane-major staging, transformed there and
//! scattered back, which reads each strided row of the block contiguously when
//! neighbouring lanes are adjacent. Kernels that transform blocks of lanes
//! across lanes instead receive whole runs, and lane pairs with equal run
//! structures receive matching source and target runs (a transposed source
//! gives one-lane runs). Traversal does not allocate and polls the
//! execution context when a pass starts and whenever the processed-lane count
//! crosses a multiple of `min(256, max(1, 65536 / len))`.
use crate::kernel::block::Run;
use scioxide_core::{ExecutionContext, SciOxideError};

/// Context polls are at most this many lanes apart.
const POLL_LANES: usize = 256;
/// Long lanes poll after about this many transformed elements.
const POLL_ELEMENTS: usize = 1 << 16;
/// Elements staged together for strided lanes, before the minimum block.
const STAGING_ELEMENTS: usize = 1 << 12;
/// Most strided lanes staged together.
const BLOCK_LANES: usize = 16;

/// How a child treats the lanes its parent hands it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneMode {
    /// The parent range-scaled finite components to at most one and owns
    /// restoration and publication; children run their unscaled kernels.
    Bounded,
    /// Every lane is an independent batch: children measure, range-scale,
    /// transform, normalize and restore it with checked arithmetic, exactly as
    /// their one-dimensional plans do.
    Checked,
}

/// Prepared lanes of one axis inside a dense row-major (sub)tensor.
///
/// `shape` and `strides` passed to traversal must be the ones used here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaneGeometry {
    axis: usize,
    /// Axis extent.
    pub(crate) len: usize,
    /// Element stride along the axis.
    pub(crate) stride: usize,
    /// Lanes per run and the start step between them.
    run: usize,
    step: usize,
    /// Extents before `split`, other than `axis`, enumerate the runs.
    split: usize,
    runs: usize,
    /// Strided lanes staged together, `pitch` elements apart: one cache line
    /// beyond the lane, so power-of-two lengths do not alias in one cache set.
    block: usize,
    pitch: usize,
}

impl LaneGeometry {
    /// Describe the lanes of `axis` over elements of type `T`. Unit extents
    /// are skipped, and trailing extents whose strides continue the innermost
    /// one densely are merged into one run.
    pub(crate) fn new<T>(shape: &[usize], strides: &[usize], axis: usize) -> Self {
        let (mut run, mut step, mut split) = (1, 0, shape.len());
        for dim in (0..shape.len()).rev() {
            if dim == axis || shape[dim] == 1 {
                continue;
            }
            if run == 1 {
                step = strides[dim];
            } else if strides[dim] != step * run {
                break;
            }
            run *= shape[dim];
            split = dim;
        }
        let runs = (0..split)
            .filter(|&dim| dim != axis)
            .map(|dim| shape[dim])
            .product();
        let len = shape[axis];
        // At least one cache line of lanes, so neighbouring lanes share the
        // strided rows they read; at most one run.
        let minimum = (64 / size_of::<T>().max(1)).clamp(1, BLOCK_LANES);
        let block = (STAGING_ELEMENTS / len.max(1))
            .clamp(minimum, BLOCK_LANES)
            .min(run);
        let pitch = len + minimum;
        Self {
            axis,
            len,
            stride: strides[axis],
            run,
            step,
            split,
            runs,
            block,
            pitch,
        }
    }

    /// Staging elements needed by [`Lanes::try_for_each`] or, for a target,
    /// by [`LanePairs::try_for_each`].
    pub(crate) const fn staging_len(&self) -> usize {
        if self.stride == 1 {
            0
        } else {
            self.block * self.pitch
        }
    }

    fn run_start(&self, mut index: usize, shape: &[usize], strides: &[usize]) -> usize {
        let mut offset = 0;
        for dim in (0..self.split).rev() {
            if dim != self.axis && shape[dim] != 1 {
                offset += (index % shape[dim]) * strides[dim];
                index /= shape[dim];
            }
        }
        offset
    }
}

/// Position of a lane geometry inside a borrowed candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement<'a> {
    pub(crate) base: usize,
    pub(crate) geometry: LaneGeometry,
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
}

impl<'a> Placement<'a> {
    /// Lanes of `geometry` starting at element zero.
    pub(crate) const fn new(
        geometry: LaneGeometry,
        shape: &'a [usize],
        strides: &'a [usize],
    ) -> Self {
        Self {
            base: 0,
            geometry,
            shape,
            strides,
        }
    }
}

impl Placement<'_> {
    fn first(&self, run: usize) -> usize {
        self.base + self.geometry.run_start(run, self.shape, self.strides)
    }
}

/// Lanes of one axis of a mutable candidate, with staging for strided lanes.
#[derive(Debug)]
pub(crate) struct Lanes<'a, T> {
    values: &'a mut [T],
    placement: Placement<'a>,
    staging: &'a mut [T],
}

impl<'a, T: Copy> Lanes<'a, T> {
    pub(crate) const fn new(
        values: &'a mut [T],
        placement: Placement<'a>,
        staging: &'a mut [T],
    ) -> Self {
        Self {
            values,
            placement,
            staging,
        }
    }

    /// Axis extent.
    pub(crate) const fn len(&self) -> usize {
        self.placement.geometry.len
    }

    /// Apply `kernel` to every lane as a contiguous slice, in lane order.
    pub(crate) fn try_for_each(
        self,
        context: ExecutionContext<'_>,
        mut kernel: impl FnMut(&mut [T]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let Self {
            values,
            placement,
            staging,
        } = self;
        let geometry = placement.geometry;
        let len = geometry.len;
        let mut poll = Poll::start(len, context)?;
        for run in 0..geometry.runs {
            let first = placement.first(run);
            let mut lane = 0;
            while lane < geometry.run {
                let start = first + lane * geometry.step;
                if geometry.stride == 1 {
                    kernel(&mut values[start..start + len])?;
                    poll.advance(context)?;
                    lane += 1;
                    continue;
                }
                let count = geometry.block.min(geometry.run - lane);
                let pitch = geometry.pitch;
                let block = &mut staging[..count * pitch];
                let shape = (geometry.step, geometry.stride, len, pitch);
                gather(values, start, shape, block);
                for staged in block.chunks_exact_mut(pitch) {
                    kernel(&mut staged[..len])?;
                    poll.advance(context)?;
                }
                scatter(block, values, start, shape);
                lane += count;
            }
        }
        Ok(())
    }

    /// Hand `kernel` runs of lanes in place, in lane order, split at context
    /// polls. Kernels address lane `j`, element `i` at `first + j * step + i * stride`.
    pub(crate) fn try_for_each_run(
        self,
        context: ExecutionContext<'_>,
        mut kernel: impl FnMut(&mut [T], Run) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        self.try_for_each_staged_run(context, |values, run, _| kernel(values, run))
    }

    /// [`Self::try_for_each_run`] that also lends the kernel the staging.
    pub(crate) fn try_for_each_staged_run(
        self,
        context: ExecutionContext<'_>,
        mut kernel: impl FnMut(&mut [T], Run, &mut [T]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let geometry = self.placement.geometry;
        let mut poll = Poll::start(geometry.len, context)?;
        for run in 0..geometry.runs {
            let first = self.placement.first(run);
            let mut lane = 0;
            while lane < geometry.run {
                let count = poll.remaining().min(geometry.run - lane);
                kernel(
                    self.values,
                    Run {
                        first: first + lane * geometry.step,
                        step: geometry.step,
                        count,
                        len: geometry.len,
                        stride: geometry.stride,
                    },
                    self.staging,
                )?;
                poll.advance_by(count, context)?;
                lane += count;
            }
        }
        Ok(())
    }
}

/// One borrowed source lane: `len` values `stride` apart from `start`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceLane<'a, T> {
    values: &'a [T],
    start: usize,
    stride: usize,
    len: usize,
}

impl<'a, T: Copy> SourceLane<'a, T> {
    /// The lane as one slice when its values are adjacent.
    pub(crate) fn as_slice(&self) -> Option<&'a [T]> {
        (self.stride == 1).then(|| &self.values[self.start..self.start + self.len])
    }

    /// The lane's values in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = T> + Clone + 'a {
        let Self {
            values,
            start,
            stride,
            len,
        } = *self;
        (0..len).map(move |i| values[start + i * stride])
    }
}

/// Lanes of one axis read from a source tensor and written to a target
/// candidate with the same remaining extents, such as a real axis and its
/// packed spectrum. Sources are read in place through [`SourceLane`], with any
/// strides; strided target lanes are staged a block at a time.
#[derive(Debug)]
pub(crate) struct LanePairs<'a, A, B> {
    source: &'a [A],
    source_placement: Placement<'a>,
    target: &'a mut [B],
    target_placement: Placement<'a>,
    staging: &'a mut [B],
}

impl<'a, A: Copy, B: Copy> LanePairs<'a, A, B> {
    pub(crate) fn new(
        (source, source_placement): (&'a [A], Placement<'a>),
        (target, target_placement): (&'a mut [B], Placement<'a>),
        staging: &'a mut [B],
    ) -> Self {
        debug_assert_eq!(
            source_placement.geometry.runs * source_placement.geometry.run,
            target_placement.geometry.runs * target_placement.geometry.run
        );
        Self {
            source,
            source_placement,
            target,
            target_placement,
            staging,
        }
    }

    /// Source and target lane lengths.
    pub(crate) const fn lens(&self) -> (usize, usize) {
        (
            self.source_placement.geometry.len,
            self.target_placement.geometry.len,
        )
    }

    /// Apply `kernel(source_lane, target_lane)` to every lane pair, in lane
    /// order.
    pub(crate) fn try_for_each(
        self,
        context: ExecutionContext<'_>,
        mut kernel: impl FnMut(SourceLane<'_, A>, &mut [B]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let Self {
            source,
            source_placement,
            target,
            target_placement,
            staging,
        } = self;
        let (from, to) = (source_placement.geometry, target_placement.geometry);
        // Equal run structures advance both sides arithmetically; otherwise
        // (for example a transposed source) each source lane is decoded.
        let aligned = (from.runs, from.run) == (to.runs, to.run);
        let source_lane = |run: usize, lane: usize| {
            let start = if aligned {
                source_placement.first(run) + lane * from.step
            } else {
                let index = run * to.run + lane;
                source_placement.first(index / from.run) + index % from.run * from.step
            };
            SourceLane {
                values: source,
                start,
                stride: from.stride,
                len: from.len,
            }
        };
        let mut poll = Poll::start(from.len.max(to.len), context)?;
        for run in 0..to.runs {
            let target_first = target_placement.first(run);
            let mut lane = 0;
            while lane < to.run {
                let start = target_first + lane * to.step;
                if to.stride == 1 {
                    kernel(source_lane(run, lane), &mut target[start..start + to.len])?;
                    poll.advance(context)?;
                    lane += 1;
                    continue;
                }
                let count = to.block.min(to.run - lane);
                let block = &mut staging[..count * to.pitch];
                for (k, output) in block.chunks_exact_mut(to.pitch).enumerate() {
                    kernel(source_lane(run, lane + k), &mut output[..to.len])?;
                    poll.advance(context)?;
                }
                scatter(block, target, start, (to.step, to.stride, to.len, to.pitch));
                lane += count;
            }
        }
        Ok(())
    }

    /// Hand `kernel` runs of lane pairs in place, in lane order, split at
    /// context polls, with the staging. Equal run structures give both sides
    /// whole runs; otherwise (for example a transposed source) every pair is
    /// a one-lane run with its source lane decoded.
    pub(crate) fn try_for_each_run(
        self,
        context: ExecutionContext<'_>,
        mut kernel: impl FnMut((&[A], Run), (&mut [B], Run), &mut [B]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let Self {
            source,
            source_placement,
            target,
            target_placement,
            staging,
        } = self;
        let (from, to) = (source_placement.geometry, target_placement.geometry);
        let aligned = (from.runs, from.run) == (to.runs, to.run);
        let lanes = |first: usize, geometry: LaneGeometry, count: usize| Run {
            first,
            step: geometry.step,
            count,
            len: geometry.len,
            stride: geometry.stride,
        };
        let mut poll = Poll::start(from.len.max(to.len), context)?;
        for run in 0..to.runs {
            let target_first = target_placement.first(run);
            let mut lane = 0;
            while lane < to.run {
                let count = if aligned {
                    poll.remaining().min(to.run - lane)
                } else {
                    1
                };
                let source_first = if aligned {
                    source_placement.first(run) + lane * from.step
                } else {
                    let index = run * to.run + lane;
                    source_placement.first(index / from.run) + index % from.run * from.step
                };
                kernel(
                    (source, lanes(source_first, from, count)),
                    (
                        &mut *target,
                        lanes(target_first + lane * to.step, to, count),
                    ),
                    &mut *staging,
                )?;
                poll.advance_by(count, context)?;
                lane += count;
            }
        }
        Ok(())
    }
}

/// Context polls at a pass start and whenever the processed-lane count
/// crosses a multiple of the interval.
struct Poll {
    interval: usize,
    done: usize,
}

impl Poll {
    fn start(len: usize, context: ExecutionContext<'_>) -> Result<Self, SciOxideError> {
        context.check()?;
        Ok(Self {
            interval: (POLL_ELEMENTS / len.max(1)).clamp(1, POLL_LANES),
            done: 0,
        })
    }

    const fn remaining(&self) -> usize {
        self.interval - self.done % self.interval
    }

    /// Count one finished lane, polling at each multiple of the interval.
    fn advance(&mut self, context: ExecutionContext<'_>) -> Result<(), SciOxideError> {
        self.advance_by(1, context)
    }

    /// Count `lanes` finished lanes that end at most at the next multiple of
    /// the interval, polling there.
    fn advance_by(
        &mut self,
        lanes: usize,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        debug_assert!(lanes <= self.remaining());
        self.done += lanes;
        if self.done.is_multiple_of(self.interval) {
            context.check()
        } else {
            Ok(())
        }
    }
}

/// Copy `block.len() / pitch` lanes of `len` values starting at `start` (lane
/// step `step`, element stride `stride`) into `block`, lane `k` at `k * pitch`,
/// one strided row at a time.
fn gather<T: Copy>(
    values: &[T],
    start: usize,
    (step, stride, len, pitch): (usize, usize, usize, usize),
    block: &mut [T],
) {
    let lanes = block.len() / pitch;
    for i in 0..len {
        let row = start + i * stride;
        for lane in 0..lanes {
            block[lane * pitch + i] = values[row + lane * step];
        }
    }
}

/// Inverse of [`gather`].
fn scatter<T: Copy>(
    block: &[T],
    values: &mut [T],
    start: usize,
    (step, stride, len, pitch): (usize, usize, usize, usize),
) {
    let lanes = block.len() / pitch;
    for i in 0..len {
        let row = start + i * stride;
        for lane in 0..lanes {
            values[row + lane * step] = block[lane * pitch + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU64, Ordering};
    use scioxide_core::{Clock, ClockBinding, Deadline, MonotonicInstant};

    fn dense(shape: &[usize]) -> Vec<usize> {
        let mut stride = 1;
        let mut strides = vec![0; shape.len()];
        for (extent, slot) in shape.iter().zip(strides.iter_mut()).rev() {
            *slot = stride;
            stride *= extent;
        }
        strides
    }

    /// Lane starts in row-major order of the other extents, by brute force.
    fn starts(shape: &[usize], strides: &[usize], axis: usize) -> Vec<usize> {
        let others: Vec<_> = (0..shape.len()).filter(|&d| d != axis).collect();
        let lanes: usize = others.iter().map(|&d| shape[d]).product();
        (0..lanes)
            .map(|mut index| {
                others.iter().rev().fold(0, |offset, &d| {
                    let coordinate = index % shape[d];
                    index /= shape[d];
                    offset + coordinate * strides[d]
                })
            })
            .collect()
    }

    #[test]
    fn runs_merge_dense_extents_and_enumerate_every_lane_in_order() {
        let cx = ExecutionContext::sequential();
        // A batch sub-tensor keeps full strides with unit extents.
        for (shape, strides) in [
            (vec![3, 4, 5], dense(&[3, 4, 5])),
            (vec![3, 1, 5, 2], dense(&[3, 7, 5, 2])),
            (vec![2, 3], vec![1, 2]),
            (vec![4], vec![1]),
        ] {
            for axis in 0..shape.len() {
                let geometry = LaneGeometry::new::<u32>(&shape, &strides, axis);
                let mut values = vec![
                    0_u32;
                    strides
                        .iter()
                        .zip(&shape)
                        .map(|(s, n)| s * n)
                        .max()
                        .unwrap()
                ];
                let mut seen = Vec::new();
                Lanes::new(
                    &mut values,
                    Placement::new(geometry, &shape, &strides),
                    &mut [],
                )
                .try_for_each_run(cx, |_, run| {
                    assert_eq!((run.len, run.stride), (shape[axis], strides[axis]));
                    seen.extend((0..run.count).map(|j| run.first + j * run.step));
                    Ok(())
                })
                .unwrap();
                assert_eq!(
                    seen,
                    starts(&shape, &strides, axis),
                    "{shape:?} axis {axis}"
                );
            }
        }
    }

    #[test]
    fn staged_lanes_and_pairs_visit_each_lane_once_in_place() {
        let cx = ExecutionContext::sequential();
        // 37 lanes leave a partial block; the source of the pair is
        // transposed, so its runs differ from the target's.
        let shape = [5, 37];
        let strides = dense(&shape);
        for axis in [0, 1] {
            let geometry = LaneGeometry::new::<f64>(&shape, &strides, axis);
            let mut values: Vec<f64> = (0..185).map(f64::from).collect();
            let original = values.clone();
            let mut staging = vec![f64::NAN; geometry.staging_len()];
            let mut lane = 0.0;
            Lanes::new(
                &mut values,
                Placement::new(geometry, &shape, &strides),
                &mut staging,
            )
            .try_for_each(cx, |values| {
                assert_eq!(values.len(), shape[axis]);
                values.iter_mut().for_each(|value| *value += 1000.0 * lane);
                lane += 1.0;
                Ok(())
            })
            .unwrap();
            for (index, start) in starts(&shape, &strides, axis).into_iter().enumerate() {
                for i in 0..shape[axis] {
                    let offset = start + i * strides[axis];
                    assert_eq!(values[offset], original[offset] + 1000.0 * index as f64);
                }
            }
            let source_strides = [1, 5];
            let source: Vec<f64> = (0..185).map(|i| f64::from(i) * 0.5).collect();
            let from = LaneGeometry::new::<f64>(&shape, &source_strides, axis);
            let mut target = vec![f64::NAN; 185];
            let mut staging = vec![f64::NAN; geometry.staging_len()];
            LanePairs::new(
                (&source, Placement::new(from, &shape, &source_strides)),
                (&mut target, Placement::new(geometry, &shape, &strides)),
                &mut staging,
            )
            .try_for_each(cx, |input, output| {
                output
                    .iter_mut()
                    .zip(input.iter())
                    .for_each(|(o, i)| *o = i);
                Ok(())
            })
            .unwrap();
            for row in 0..5 {
                for column in 0..37 {
                    assert_eq!(target[row * 37 + column], source[row + 5 * column]);
                }
            }
        }
    }

    #[test]
    fn pair_runs_visit_every_pair_in_lane_order() {
        let cx = ExecutionContext::sequential();
        // An aligned source shares the target's runs; a transposed source
        // (different run structure) gives one-lane runs. Both visit the pairs
        // of `try_for_each` in the same order.
        let shape = [3, 4, 5];
        let target_strides = dense(&shape);
        for source_strides in [dense(&shape), vec![1, 3, 12]] {
            for axis in 0..3 {
                let from = LaneGeometry::new::<u32>(&shape, &source_strides, axis);
                let to = LaneGeometry::new::<u32>(&shape, &target_strides, axis);
                let source = vec![0_u32; 60];
                let mut target = vec![0_u32; 60];
                let mut staging = vec![0_u32; to.staging_len()];
                let mut runs = Vec::new();
                LanePairs::new(
                    (&source, Placement::new(from, &shape, &source_strides)),
                    (&mut target, Placement::new(to, &shape, &target_strides)),
                    &mut staging,
                )
                .try_for_each_run(cx, |(_, source), (_, target), _| {
                    assert_eq!(source.count, target.count);
                    for lane in 0..source.count {
                        runs.push((
                            source.first + lane * source.step,
                            target.first + lane * target.step,
                        ));
                    }
                    Ok(())
                })
                .unwrap();
                let expected: Vec<_> = starts(&shape, &source_strides, axis)
                    .into_iter()
                    .zip(starts(&shape, &target_strides, axis))
                    .collect();
                assert_eq!(runs, expected, "{source_strides:?} axis {axis}");
            }
        }
    }

    struct Counter(AtomicU64);
    impl Clock for Counter {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::ZERO,
            ))
        }
    }

    #[test]
    fn passes_poll_at_entry_and_at_most_256_lanes_apart() {
        // Short lanes poll every 256 lanes; long lanes after about 65536
        // elements. Polls fire when the processed-lane count crosses a
        // multiple of the interval, also inside staged blocks and runs.
        for (len, lanes, polls) in [(2, 600, 3), (1024, 600, 10), (65536, 3, 4)] {
            let shape = [len, lanes];
            let strides = dense(&shape);
            let clock = Counter(AtomicU64::new(0));
            let binding = ClockBinding::new(&clock);
            let deadline = MonotonicInstant::from_duration_since_epoch(core::time::Duration::MAX);
            let cx = ExecutionContext::sequential()
                .with_deadline(Deadline::at(&binding, deadline))
                .unwrap();
            let geometry = LaneGeometry::new::<u8>(&shape, &strides, 0);
            let mut values = vec![0_u8; len * lanes];
            let mut staging = vec![0_u8; geometry.staging_len()];
            Lanes::new(
                &mut values,
                Placement::new(geometry, &shape, &strides),
                &mut staging,
            )
            .try_for_each(cx, |_| Ok(()))
            .unwrap();
            assert_eq!(clock.0.load(Ordering::Relaxed), polls, "len {len}");
            clock.0.store(0, Ordering::Relaxed);
            Lanes::new(
                &mut values,
                Placement::new(geometry, &shape, &strides),
                &mut staging,
            )
            .try_for_each_run(cx, |_, _| Ok(()))
            .unwrap();
            assert_eq!(clock.0.load(Ordering::Relaxed), polls, "len {len} runs");
        }
    }
}
