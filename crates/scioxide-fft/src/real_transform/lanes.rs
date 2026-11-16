//! N-D trigonometric lane drivers and block passes.
use super::*;
use crate::kernel::block::{DigitMajor, LaneMajor, LaneRead, LaneWrite, PermutedRows, RealLanes};

/// Native blocks read either their candidate or a separate borrowed run.
enum NativeSource<'a, S> {
    InPlace,
    Borrowed(&'a [S], Run),
}

impl<S: Copy> NativeSource<'_, S> {
    #[inline(always)]
    fn read<'a, A: Addressing>(
        &'a self,
        target: &'a [S],
        run: Run,
        block: usize,
    ) -> A::Read<'a, S> {
        let (values, from) = match self {
            Self::InPlace => (target, run),
            Self::Borrowed(values, from) => (*values, *from),
        };
        A::read(values, from, block)
    }

    /// Copy only the next fallback lane, keeping lane-ordered failures intact.
    fn prepare_lane(&self, target: &mut [S], to: Run, lane: usize) {
        if let Self::Borrowed(source, from) = self {
            for i in 0..to.len {
                target[to.first + lane * to.step + i * to.stride] =
                    source[from.first + lane * from.step + i * from.stride];
            }
        }
    }
}

impl<S: scioxide_core::RealScalar> RealTransformPlan<S> {
    /// Whether native blocks can consume a borrowed source directly.
    pub(crate) fn native_lanes(&self) -> bool {
        self.blocks.is_some() && self.even_child().0.smooth_stages().is_some()
    }

    /// Independent checked lanes read directly from caller storage. Only the
    /// parent's candidate is written, and exceptional blocks keep lane order.
    pub(crate) fn transform_source_lanes(
        &self,
        pairs: crate::lanes::LanePairs<'_, S, S>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::sufficient(scratch.len(), self.lane_scratch_len())?;
        pairs.try_for_each_run(context, |source, target, _| match Layout::of(source.1) {
            Layout::Adjacent => self.source_run::<Adjacent>(source, target, scratch, context),
            Layout::Contiguous => self.source_run::<Contiguous>(source, target, scratch, context),
            Layout::Spread => self.source_run::<Spread>(source, target, scratch, context),
        })
    }

    fn source_run<A: Addressing>(
        &self,
        (source, from): (&[S], Run),
        (target, to): (&mut [S], Run),
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let blocks = self
            .blocks
            .unwrap_or_else(|| unreachable!("native lane plan"));
        crate::validation::sufficient(scratch.len(), blocks.scratch_len)?;
        let mut rows = [Block::new(Reals::splat(S::ZERO), Reals::splat(S::ZERO)); ROWS];
        self.native_blocks::<A>(
            (NativeSource::Borrowed(source, from), target, to),
            (&mut rows, scratch),
            blocks.dispatch,
            LaneMode::Checked,
            context,
        )
    }

    fn publish_native<R: Load<Block<S, WIDTH>>>(
        &self,
        source: R,
        (values, run): (&mut [S], Run),
        block: usize,
        restore: Reals<S, WIDTH>,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) {
        match Layout::of(run) {
            Layout::Adjacent => crate::kernel::run_pass(
                dispatch,
                DirectPost {
                    plan: self,
                    source,
                    target: Adjacent::write(values, run, block),
                    restore,
                },
            ),
            Layout::Contiguous => crate::kernel::run_pass(
                dispatch,
                DirectPost {
                    plan: self,
                    source,
                    target: Contiguous::write(values, run, block),
                    restore,
                },
            ),
            Layout::Spread => crate::kernel::run_pass(
                dispatch,
                DirectPost {
                    plan: self,
                    source,
                    target: Spread::write(values, run, block),
                    restore,
                },
            ),
        }
    }

    /// Child scratch of an N-D lane pass, which needs no candidate storage.
    pub(crate) fn lane_scratch_len(&self) -> usize {
        self.blocks
            .map_or(self.scratch_len, |blocks| blocks.scratch_len)
    }

    /// Real staging of an N-D lane pass: one interleaved block of lanes.
    pub(crate) fn lane_staging_len(&self) -> usize {
        self.blocks.map_or(0, |_| {
            if self.even_child().0.smooth_stages().is_some() {
                0
            } else {
                WIDTH * self.spec.length
            }
        })
    }

    /// Transform every lane of one N-D axis in place on behalf of the parent,
    /// with one length and scratch check for the whole axis. The lanes belong
    /// to the parent's candidate, so outputs are written straight into them.
    ///
    /// Bounded lanes are finite and at most one in the parent's frame and use
    /// the fixed exponent two; checked lanes are independent batches with the
    /// arithmetic of [`Self::execute_into`]. Short lanes run in place across
    /// whole runs. Even II/III/IV lanes run in blocks of [`WIDTH`]; see
    /// [`Self::transform_blocks`].
    pub(crate) fn transform_lanes(
        &self,
        lanes: Lanes<'_, S>,
        scratch: &mut [Complex<S>],
        mode: LaneMode,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.spec.length;
        crate::validation::length(lanes.len(), n)?;
        crate::validation::sufficient(scratch.len(), self.lane_scratch_len())?;
        if let Some(short) = self.short {
            return lanes
                .try_for_each_run(context, |values, run| short.apply_run(values, run, mode));
        }
        if let Some(blocks) = self.blocks {
            let rows = &mut [Block::new(Reals::splat(S::ZERO), Reals::splat(S::ZERO)); ROWS];
            return lanes.try_for_each_staged_run(context, |values, run, staging| {
                let staging = &mut staging[..self.lane_staging_len()];
                let scratch = &mut scratch[..blocks.scratch_len];
                match Layout::of(run) {
                    Layout::Adjacent => self.transform_blocks::<Adjacent>(
                        values,
                        run,
                        (staging, rows, scratch),
                        blocks.dispatch,
                        mode,
                        context,
                    ),
                    Layout::Contiguous => self.transform_blocks::<Contiguous>(
                        values,
                        run,
                        (staging, rows, scratch),
                        blocks.dispatch,
                        mode,
                        context,
                    ),
                    Layout::Spread => self.transform_blocks::<Spread>(
                        values,
                        run,
                        (staging, rows, scratch),
                        blocks.dispatch,
                        mode,
                        context,
                    ),
                }
            });
        }
        let work = &mut scratch[n..];
        lanes.try_for_each(context, |lane| {
            let exponent = match mode {
                LaneMode::Bounded => 2,
                LaneMode::Checked => {
                    let Some(exponent) = lane_exponent(lane)? else {
                        lane.fill(S::ZERO);
                        return Ok(());
                    };
                    exponent
                }
            };
            let raw = self.raw_spectrum(lane, work, BinaryScale::new(-exponent), context)?;
            let restore = BinaryScale::new(exponent);
            self.publish(
                |k| Block::from(raw[k]),
                |i, value| lane[i] = restore.restore(value.0[0]),
            );
            if !self.output_bound.scalbn(exponent).is_finite()
                && lane.iter().any(|value| !value.is_finite())
            {
                return Err(overflow());
            }
            Ok(())
        })
    }

    /// Transform one run of an even II/III/IV axis whose lanes are addressed
    /// by `A`, a block of [`WIDTH`] lanes at a time.
    ///
    /// Smooth children stage directly in native digit order and write directly
    /// to the candidate. Other children keep interleaved block adapters around
    /// individual calls. Checked blocks require finite input, ordinary binary
    /// scales and proven-finite restoration; exceptional and partial native
    /// blocks execute alone in lane order, preserving arithmetic and errors.
    pub(super) fn transform_blocks<A: Addressing>(
        &self,
        values: &mut [S],
        run: Run,
        (lanes, rows, scratch): (&mut [S], &mut [Block<S, WIDTH>; ROWS], &mut [Complex<S>]),
        dispatch: scioxide_core::backend::KernelDispatch,
        mode: LaneMode,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        // Bounded lanes use the fixed exponent two.
        if self.even_child().0.smooth_stages().is_some() {
            return self.native_blocks::<A>(
                (NativeSource::InPlace, values, run),
                (rows, scratch),
                dispatch,
                mode,
                context,
            );
        }
        let four = S::TWO + S::TWO;
        let bounded = (Reals::splat(S::ONE / four), Reals::splat(four));
        for block in (0..run.count).step_by(WIDTH) {
            let count = (run.count - block).min(WIDTH);
            if count == WIDTH {
                interleave::<A, S>(values, run, block, lanes);
            } else {
                interleave_partial(values, (run, block, count), lanes, S::ZERO);
            }
            let scales = match mode {
                LaneMode::Bounded => Some(bounded),
                LaneMode::Checked => self.block_scales(&Interleaved(&*lanes)),
            };
            let Some((down, restore)) = scales else {
                for lane in block..block + count {
                    self.transform_lane(values, run, lane, scratch, mode, context)?;
                }
                continue;
            };
            crate::kernel::run_pass(
                dispatch,
                StageRows {
                    plan: self,
                    lanes,
                    rows,
                    down,
                },
            );
            self.lane_children(rows, scratch, context)?;
            crate::kernel::run_pass(
                dispatch,
                PostRows {
                    plan: self,
                    rows,
                    lanes,
                    restore,
                },
            );
            if count == WIDTH {
                deinterleave::<A, S>(lanes, values, run, block);
            } else {
                deinterleave_partial(lanes, values, (run, block, count));
            }
        }
        Ok(())
    }

    /// Stage blocks from either source into native child storage, then write
    /// restored outputs directly to the candidate. Partial and exceptional
    /// blocks use individual lanes in source order.
    fn native_blocks<A: Addressing>(
        &self,
        (source, values, run): (NativeSource<'_, S>, &mut [S], Run),
        (rows, scratch): (&mut [Block<S, WIDTH>; ROWS], &mut [Complex<S>]),
        dispatch: scioxide_core::backend::KernelDispatch,
        mode: LaneMode,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (child, inverse) = self.even_child();
        let stages = child
            .smooth_stages()
            .unwrap_or_else(|| unreachable!("native smooth block"));
        let half = self.spec.length / 2;
        for block in (0..run.count).step_by(WIDTH) {
            let count = (run.count - block).min(WIDTH);
            let mut staged = false;
            let scales = if count < WIDTH {
                None
            } else if mode == LaneMode::Checked
                && self.raw_type() == KernelType::Two
                && stages.native_rows_supported()
            {
                crate::kernel::run_pass(
                    dispatch,
                    MeasuredTwo {
                        plan: self,
                        source: source.read::<A>(values, run, block),
                        rows,
                    },
                )
                .map(|(down, restore, ready)| {
                    staged = ready;
                    (down, restore)
                })
            } else {
                match mode {
                    LaneMode::Bounded => {
                        let four = S::TWO + S::TWO;
                        Some((Reals::splat(S::ONE / four), Reals::splat(four)))
                    }
                    LaneMode::Checked => crate::kernel::run_pass(
                        dispatch,
                        DirectScales {
                            plan: self,
                            source: source.read::<A>(values, run, block),
                        },
                    ),
                }
            };
            let Some((down, restore)) = scales else {
                for lane in block..block + count {
                    source.prepare_lane(values, run, lane);
                    self.transform_lane(values, run, lane, scratch, mode, context)?;
                }
                continue;
            };
            if stages.native_rows_supported() {
                if !staged {
                    crate::kernel::run_pass(
                        dispatch,
                        DirectStage {
                            plan: self,
                            source: source.read::<A>(values, run, block),
                            destination: PermutedRows {
                                rows,
                                order: &self.digit_rows,
                            },
                            down,
                        },
                    );
                }
                stages.transform_native_rows(rows, inverse);
                self.publish_native(&rows[..half], (values, run), block, restore, dispatch);
                continue;
            }
            let (data, work) = scratch.split_at_mut(WIDTH * half);
            crate::kernel::run_pass(
                dispatch,
                DirectStage {
                    plan: self,
                    source: source.read::<A>(values, run, block),
                    destination: DigitMajor {
                        values: data,
                        order: &self.digit_rows,
                    },
                    down,
                },
            );
            stages.transform_staged_batch(data, work, inverse, S::ONE);
            self.publish_native(
                LaneMajor {
                    values: &*work,
                    len: half,
                },
                (values, run),
                block,
                restore,
                dispatch,
            );
        }
        Ok(())
    }

    /// Run the half-length child on each lane of natural-order `rows` through
    /// the lane's own contiguous staging row, as one-dimensional execution
    /// does.
    pub(super) fn lane_children(
        &self,
        rows: &mut [Block<S, WIDTH>; ROWS],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let half = self.spec.length / 2;
        let (child, inverse) = self.even_child();
        let (row, work) = scratch.split_at_mut(half);
        for lane in 0..WIDTH {
            for (slot, value) in row.iter_mut().zip(&rows[..half]) {
                *slot = value.lane(lane);
            }
            child.execute_bounded(row, work, inverse, S::ONE, context)?;
            for (value, slot) in rows[..half].iter_mut().zip(&*row) {
                value.re.0[lane] = slot.re;
                value.im.0[lane] = slot.im;
            }
        }
        Ok(())
    }

    /// Per-lane down and restore factors of an interleaved checked block, or
    /// `None` unless every lane may run in the block: finite, with ordinary
    /// binary scales and a provably finite restoration. All-zero lanes,
    /// including the padding of a partial block, keep unit scales; their
    /// outputs restore to positive zeros.
    #[inline(always)]
    pub(super) fn block_scales(
        &self,
        lanes: &impl Load<Reals<S, WIDTH>>,
    ) -> Option<(Reals<S, WIDTH>, Reals<S, WIDTH>)> {
        let mut maximum = Reals::<S, WIDTH>::splat(S::ZERO);
        let mut finite = true;
        for i in 0..self.spec.length {
            let value = lanes.load(i);
            finite &= value.finite();
            maximum = maximum.max_abs(value);
        }
        if !finite {
            return None;
        }
        self.scales_for_maximum(maximum)
    }

    #[inline(always)]
    fn scales_for_maximum(
        &self,
        maximum: Reals<S, WIDTH>,
    ) -> Option<(Reals<S, WIDTH>, Reals<S, WIDTH>)> {
        let (mut down, mut restore) = ([S::ONE; WIDTH], [S::ONE; WIDTH]);
        // Neighbouring lanes usually share a binary range.
        let mut previous = None;
        for lane in 0..WIDTH {
            if maximum.0[lane] == S::ZERO {
                continue;
            }
            let exponent = crate::scaling::range_exponent(maximum.0[lane]) + 2;
            let scales = match previous {
                Some((cached, scales)) if cached == exponent => scales,
                _ => {
                    if !self.output_bound.scalbn(exponent).is_finite() {
                        return None;
                    }
                    let scales = (
                        BinaryScale::new(-exponent).factor()?,
                        BinaryScale::new(exponent).factor()?,
                    );
                    previous = Some((exponent, scales));
                    scales
                }
            };
            (down[lane], restore[lane]) = scales;
        }
        Some((Reals(down), Reals(restore)))
    }

    /// Transform lane `lane` of `run` in place, alone, with the arithmetic of
    /// the slice lanes of [`Self::transform_lanes`].
    pub(super) fn transform_lane(
        &self,
        values: &mut [S],
        run: Run,
        lane: usize,
        scratch: &mut [Complex<S>],
        mode: LaneMode,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.spec.length;
        let start = run.first + lane * run.step;
        let at = |i: usize| start + i * run.stride;
        let exponent = match mode {
            LaneMode::Bounded => 2,
            LaneMode::Checked => {
                let maximum = crate::scaling::real_max((0..n).map(|i| values[at(i)]))?;
                if maximum == S::ZERO {
                    (0..n).for_each(|i| values[at(i)] = S::ZERO);
                    return Ok(());
                }
                crate::scaling::range_exponent(maximum) + 2
            }
        };
        let (row, work) = scratch.split_at_mut(n / 2);
        let raw = self.even_spectrum(
            |i| values[at(i)],
            BinaryScale::new(-exponent),
            row,
            work,
            context,
        )?;
        let restore = BinaryScale::new(exponent);
        self.publish(
            |k| Block::from(raw[k]),
            |i, value| values[at(i)] = restore.restore(value.0[0]),
        );
        if !self.output_bound.scalbn(exponent).is_finite()
            && (0..n).any(|i| !values[at(i)].is_finite())
        {
            return Err(overflow());
        }
        Ok(())
    }
}

/// Try the ordinary exponent-two frame while measuring II's unweighted
/// pairs. A mismatched range is restaged with its measured factors; no child
/// runs and no candidate is written until the checked scales are available.
struct MeasuredTwo<'p, 'r, S: scioxide_core::RealScalar, V> {
    plan: &'p RealTransformPlan<S>,
    source: V,
    rows: &'r mut [Block<S, WIDTH>; ROWS],
}
impl<S: scioxide_core::RealScalar, V: LaneRead<S>> crate::kernel::block::BlockPass
    for MeasuredTwo<'_, '_, S, V>
{
    type Output = Option<(Reals<S, WIDTH>, Reals<S, WIDTH>, bool)>;
    #[inline(always)]
    fn run(self) -> Self::Output {
        let quarter = S::ONE / (S::TWO + S::TWO);
        let mut output = MeasuredPairs {
            output: PermutedRows {
                rows: self.rows,
                order: &self.plan.digit_rows,
            },
            maximum: Reals::splat(S::ZERO),
            finite: true,
            quarter,
        };
        staging::stage_even_two(
            &mut output,
            self.plan.spec.length,
            &RealLanes(self.source),
            if matches!(self.plan.spec.kind, RealTransformKind::Dst(_)) {
                -S::ONE
            } else {
                S::ONE
            },
        );
        if !output.finite {
            return None;
        }
        let (down, restore) = self.plan.scales_for_maximum(output.maximum)?;
        // Multiplication by a positive factor leaves either zero's bits alone.
        let ready =
            (0..WIDTH).all(|lane| output.maximum.0[lane] == S::ZERO || down.0[lane] == quarter);
        Some((down, restore, ready))
    }
}

struct MeasuredPairs<'a, S: scioxide_core::RealScalar> {
    output: PermutedRows<'a, S>,
    maximum: Reals<S, WIDTH>,
    finite: bool,
    quarter: S,
}
impl<S: scioxide_core::RealScalar> Store<Block<S, WIDTH>> for MeasuredPairs<'_, S> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        self.maximum = self.maximum.max_abs(value.re).max_abs(value.im);
        self.finite &= value.re.finite() & value.im.finite();
        self.output.store(
            i,
            Block::new(value.re * self.quarter, value.im * self.quarter),
        );
    }
}

struct DirectScales<'p, S: scioxide_core::RealScalar, V> {
    plan: &'p RealTransformPlan<S>,
    source: V,
}

impl<S: scioxide_core::RealScalar, V: LaneRead<S>> crate::kernel::block::BlockPass
    for DirectScales<'_, S, V>
{
    type Output = Option<(Reals<S, WIDTH>, Reals<S, WIDTH>)>;
    #[inline(always)]
    fn run(self) -> Self::Output {
        self.plan.block_scales(&RealLanes(self.source))
    }
}

struct DirectStage<'p, S: scioxide_core::RealScalar, V, W> {
    plan: &'p RealTransformPlan<S>,
    source: V,
    destination: W,
    down: Reals<S, WIDTH>,
}

impl<S: scioxide_core::RealScalar, V: LaneRead<S>, W: Store<Block<S, WIDTH>>>
    crate::kernel::block::BlockPass for DirectStage<'_, S, V, W>
{
    type Output = ();
    #[inline(always)]
    fn run(mut self) {
        self.plan
            .stage_even(&RealLanes(self.source), &self.down, &mut self.destination);
    }
}

struct DirectPost<'p, S: scioxide_core::RealScalar, R, V> {
    plan: &'p RealTransformPlan<S>,
    source: R,
    target: V,
    restore: Reals<S, WIDTH>,
}

impl<S: scioxide_core::RealScalar, R: Load<Block<S, WIDTH>>, V: LaneWrite<S>>
    crate::kernel::block::BlockPass for DirectPost<'_, S, R, V>
{
    type Output = ();
    #[inline(always)]
    fn run(self) {
        self.plan.publish_even(
            &self.source,
            &mut Restored {
                lanes: RealLanes(self.target),
                restore: self.restore,
            },
        );
    }
}
/// Stage one nonsmooth block in natural order for individual child calls.
struct StageRows<'p, 'r, S: scioxide_core::RealScalar> {
    plan: &'p RealTransformPlan<S>,
    lanes: &'r [S],
    rows: &'r mut [Block<S, WIDTH>; ROWS],
    down: Reals<S, WIDTH>,
}

impl<S: scioxide_core::RealScalar> crate::kernel::block::BlockPass for StageRows<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        let lanes = Interleaved(self.lanes);
        self.plan
            .stage_even(&lanes, &self.down, &mut &mut self.rows[..]);
    }
}

/// Post-processing of one block's rows into its interleaved lanes.
struct PostRows<'p, 'r, S: scioxide_core::RealScalar> {
    plan: &'p RealTransformPlan<S>,
    rows: &'r [Block<S, WIDTH>; ROWS],
    lanes: &'r mut [S],
    restore: Reals<S, WIDTH>,
}

impl<S: scioxide_core::RealScalar> crate::kernel::block::BlockPass for PostRows<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        self.plan.publish_even(
            &&self.rows[..],
            &mut Restored {
                lanes: Interleaved(self.lanes),
                restore: self.restore,
            },
        );
    }
}

/// Block outputs restored with each lane's binary factor and stored in place.
struct Restored<W, S> {
    lanes: W,
    restore: Reals<S, WIDTH>,
}

impl<S: scioxide_core::RealScalar, W: Store<Reals<S, WIDTH>>> Store<Reals<S, WIDTH>>
    for Restored<W, S>
{
    #[inline(always)]
    fn store(&mut self, i: usize, value: Reals<S, WIDTH>) {
        self.lanes.store(i, value.restore(self.restore));
    }
}
