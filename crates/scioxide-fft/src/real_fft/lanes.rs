//! N-D bounded and checked real FFT lane drivers.
use super::*;
#[cfg(test)]
use crate::kernel::block::Interleaved;
use crate::kernel::block::{
    Addressing, Adjacent, Contiguous, DigitMajor, LaneMajor, LaneRead, LaneWrite, Layout,
    PermutedRows, Spread,
};
use crate::kernel::scalar::Butterfly;

impl<S: scioxide_core::RealScalar> RealFftPlan<S> {
    /// Forward N-D real source lanes into packed lanes on behalf of the
    /// parent, with one length and scratch check for the axis. Bounded sources
    /// are staged scaled by `down`, the parent's batch scale, and produce
    /// unscaled bins; checked lanes follow the one-dimensional view arithmetic
    /// with normalization `scale`.
    pub(crate) fn forward_lanes(
        &self,
        pairs: LanePairs<'_, S, Complex<S>>,
        scratch: &mut [Complex<S>],
        mode: LaneMode,
        (down, scale): (BinaryScale<S>, S),
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (samples, bins) = pairs.lens();
        crate::validation::length(samples, self.n)?;
        crate::validation::length(bins, self.spectrum_len)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        let prechecked = output_factor(self.output_bound(scale), 0, scale).is_some();
        if let (LaneMode::Bounded, Some(dispatch), Some(down), true) =
            (mode, self.blocks, down.factor(), prechecked)
        {
            crate::validation::sufficient(scratch.len(), self.lane_scratch_len())?;
            return pairs.try_for_each_run(context, |source, target, _| {
                NativeReal {
                    plan: self,
                    source: NativeSource::Real(source),
                    target,
                    scratch,
                    down,
                    scale,
                }
                .execute(dispatch);
                Ok(())
            });
        }
        pairs.try_for_each(context, |input, output| {
            let exponent = match (mode, input.as_slice()) {
                (LaneMode::Bounded, samples) => {
                    match samples {
                        Some(samples) => self.stage_slice(scratch, samples, down),
                        None => self.stage_samples(scratch, input.iter(), down),
                    }
                    self.transform_staged(scratch, context)?;
                    Some(0)
                }
                (LaneMode::Checked, Some(samples)) => {
                    self.forward_slice(samples, scratch, context)?
                }
                (LaneMode::Checked, None) => {
                    self.forward_scaled(|| input.iter(), scratch, context)?
                }
            };
            self.publish_bins(scale, S::ONE, scratch, exponent, Bins::Contiguous(output))
        })
    }

    /// Invert packed N-D source lanes into the real parts of target lanes on
    /// behalf of the parent, with one length and scratch check for the axis.
    /// Bounded lanes are projected spectra with finite components at most one
    /// and give unscaled samples; checked lanes follow the view arithmetic,
    /// with imaginary parts times `sign`, endpoint projection and
    /// normalization `scale`.
    pub(crate) fn inverse_lanes(
        &self,
        pairs: LanePairs<'_, Complex<S>, Complex<S>>,
        scratch: &mut [Complex<S>],
        mode: LaneMode,
        (sign, scale): (S, S),
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (bins, samples) = pairs.lens();
        crate::validation::length(bins, self.spectrum_len)?;
        crate::validation::length(samples, self.n)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        let prechecked = output_factor(self.output_bound(S::ONE), 3, S::ONE).is_some();
        if let (LaneMode::Bounded, Some(dispatch), true) = (mode, self.blocks, prechecked) {
            crate::validation::sufficient(scratch.len(), self.lane_scratch_len())?;
            return pairs.try_for_each_run(context, |source, target, _| {
                NativeReal {
                    plan: self,
                    source: NativeSource::Packed(source),
                    target,
                    scratch,
                    down: S::ONE,
                    scale: S::ONE,
                }
                .execute(dispatch);
                Ok(())
            });
        }
        pairs.try_for_each(context, |input, output| {
            let (factor, restore) = match (mode, input.as_slice()) {
                (LaneMode::Bounded, _) => {
                    for (slot, value) in scratch[..self.spectrum_len].iter_mut().zip(input.iter()) {
                        *slot = crate::validation::bounded(value);
                    }
                    assert!(
                        scratch[0].im == S::ZERO,
                        "bounded inverse requires projected DC"
                    );
                    assert!(
                        !self.n.is_multiple_of(2) || scratch[self.n / 2].im == S::ZERO,
                        "bounded inverse requires projected Nyquist"
                    );
                    self.inverse_core(S::ONE, scratch, context, 2)?
                }
                (LaneMode::Checked, Some(bins)) => {
                    self.inverse_bins(scale, bins.iter().copied(), sign, scratch, context)?
                }
                (LaneMode::Checked, None) => {
                    self.inverse_bins(scale, input.iter(), sign, scratch, context)?
                }
            };
            self.publish_samples(
                &scratch[..self.staged_len()],
                factor,
                restore,
                output.iter_mut().map(|slot| {
                    slot.im = S::ZERO;
                    &mut slot.re
                }),
            );
            Ok(())
        })
    }
}
#[derive(Clone, Copy)]
enum NativeSource<'a, S> {
    Real((&'a [S], Run)),
    Packed((&'a [Complex<S>], Run)),
}

struct NativeReal<'p, 'r, S: scioxide_core::RealScalar> {
    plan: &'p RealFftPlan<S>,
    source: NativeSource<'r, S>,
    target: (&'r mut [Complex<S>], Run),
    scratch: &'r mut [Complex<S>],
    down: S,
    scale: S,
}

impl<S: scioxide_core::RealScalar> NativeReal<'_, '_, S> {
    fn execute(self, dispatch: scioxide_core::backend::KernelDispatch) {
        let from = match self.source {
            NativeSource::Real((_, from)) | NativeSource::Packed((_, from)) => from,
        };
        match Layout::of(from) {
            Layout::Adjacent => self.run_from::<Adjacent>(dispatch),
            Layout::Contiguous => self.run_from::<Contiguous>(dispatch),
            Layout::Spread => self.run_from::<Spread>(dispatch),
        }
    }
    fn run_from<A: Addressing>(self, dispatch: scioxide_core::backend::KernelDispatch) {
        match Layout::of(self.target.1) {
            Layout::Adjacent => crate::kernel::run_pass(
                dispatch,
                AddressedReal::<_, A, Adjacent>(self, core::marker::PhantomData),
            ),
            Layout::Contiguous => crate::kernel::run_pass(
                dispatch,
                AddressedReal::<_, A, Contiguous>(self, core::marker::PhantomData),
            ),
            Layout::Spread => crate::kernel::run_pass(
                dispatch,
                AddressedReal::<_, A, Spread>(self, core::marker::PhantomData),
            ),
        }
    }

    /// Select source and target addressing once per run. Tails use the same
    /// pipeline with repeated finite lanes and write only their real lanes.
    #[inline(always)]
    fn run_layout<A: Addressing, B: Addressing>(self, rows: &mut [Block<S, WIDTH>; ROWS]) {
        let (child, twiddles) = self
            .plan
            .even_parts()
            .unwrap_or_else(|| unreachable!("even native block"));
        let stages = child
            .smooth_stages()
            .unwrap_or_else(|| unreachable!("smooth native child"));
        let (target, to) = self.target;
        let half = self.plan.n / 2;
        // Axis admission established these lengths. Express them through the
        // same half length used by staging and publication, so full-row
        // accessors can eliminate their inner component bounds checks.
        let to = Run {
            len: match self.source {
                NativeSource::Real(_) => half + 1,
                NativeSource::Packed(_) => 2 * half,
            },
            ..to
        };
        for block in (0..to.count).step_by(WIDTH) {
            let buffers = (&mut *rows, &mut *self.scratch);
            if to.count - block >= WIDTH {
                let output = B::write(target, to, block);
                match self.source {
                    NativeSource::Real((source, from)) => native_block(
                        self.plan,
                        RealInput(A::read(
                            source,
                            Run {
                                len: 2 * half,
                                ..from
                            },
                            block,
                        )),
                        output,
                        buffers,
                        stages,
                        twiddles,
                        (self.down, self.scale),
                    ),
                    NativeSource::Packed((source, from)) => native_block(
                        self.plan,
                        PackedInput {
                            lanes: A::read(
                                source,
                                Run {
                                    len: half + 1,
                                    ..from
                                },
                                block,
                            ),
                            half,
                        },
                        output,
                        buffers,
                        stages,
                        twiddles,
                        (self.down, self.scale),
                    ),
                }
            } else {
                let output = PartialLanes {
                    values: &mut *target,
                    lanes: BlockLanes::new(to, block),
                };
                match self.source {
                    NativeSource::Real((source, from)) => native_block(
                        self.plan,
                        RealInput(PartialLanes {
                            values: source,
                            lanes: BlockLanes::new(from, block),
                        }),
                        output,
                        buffers,
                        stages,
                        twiddles,
                        (self.down, self.scale),
                    ),
                    NativeSource::Packed((source, from)) => native_block(
                        self.plan,
                        PackedInput {
                            lanes: PartialLanes {
                                values: source,
                                lanes: BlockLanes::new(from, block),
                            },
                            half: self.plan.n / 2,
                        },
                        output,
                        buffers,
                        stages,
                        twiddles,
                        (self.down, self.scale),
                    ),
                }
            }
        }
    }
}

/// Layout selection precedes ISA dispatch, keeping each compiled pass small.
struct AddressedReal<P, A, B>(P, core::marker::PhantomData<(A, B)>);
impl<S: scioxide_core::RealScalar, A: Addressing, B: Addressing> crate::kernel::block::BlockPass
    for AddressedReal<NativeReal<'_, '_, S>, A, B>
{
    type Output = ();
    #[inline(always)]
    fn run(self) {
        self.0.run_layout::<A, B>(&mut [Block::zero(); ROWS]);
    }
}

/// A source supplies either packed samples or the inverse's projected bins.
trait NativeInput<S: scioxide_core::RealScalar> {
    const INVERSE: bool;
    fn stage(
        &self,
        half: usize,
        twiddles: &[Complex<S>],
        down: S,
        destination: &mut impl Store<Block<S, WIDTH>>,
    );
}

struct RealInput<V>(V);
impl<S: scioxide_core::RealScalar, V: LaneRead<S>> NativeInput<S> for RealInput<V> {
    const INVERSE: bool = false;
    #[inline(always)]
    fn stage(
        &self,
        half: usize,
        _twiddles: &[Complex<S>],
        down: S,
        destination: &mut impl Store<Block<S, WIDTH>>,
    ) {
        for i in 0..half {
            let even = Reals(self.0.read(2 * i));
            let odd = Reals(self.0.read(2 * i + 1));
            destination.store(i, Block::new(even * down, odd * down));
        }
    }
}

struct PackedInput<V> {
    lanes: V,
    half: usize,
}
impl<S: scioxide_core::RealScalar, V: LaneRead<Complex<S>>> Load<Block<S, WIDTH>>
    for PackedInput<V>
{
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, WIDTH> {
        let input = self.lanes.read(i);
        let value = Block::from_fn(|lane| input[lane]).bounded();
        assert!(
            i != 0 && i != self.half || value.im.0.into_iter().all(|v| v == S::ZERO),
            "bounded inverse requires projected endpoints"
        );
        value * (S::ONE / (S::TWO + S::TWO))
    }
}
impl<S: scioxide_core::RealScalar, V: LaneRead<Complex<S>>> NativeInput<S> for PackedInput<V> {
    const INVERSE: bool = true;
    #[inline(always)]
    fn stage(
        &self,
        half: usize,
        twiddles: &[Complex<S>],
        _down: S,
        destination: &mut impl Store<Block<S, WIDTH>>,
    ) {
        even::merge_even(half, twiddles, self, destination);
    }
}

/// Both child storage layouts share staging and publication arithmetic.
#[inline(always)]
fn native_block<S, I, O>(
    plan: &RealFftPlan<S>,
    input: I,
    output: O,
    (rows, scratch): (&mut [Block<S, WIDTH>; ROWS], &mut [Complex<S>]),
    stages: &crate::kernel::StagePlan<S>,
    twiddles: &[Complex<S>],
    (down, scale): (S, S),
) where
    S: scioxide_core::RealScalar,
    I: NativeInput<S>,
    O: LaneWrite<Complex<S>>,
{
    let half = plan.n / 2;
    if stages.native_rows_supported() {
        input.stage(
            half,
            twiddles,
            down,
            &mut PermutedRows {
                rows,
                order: &plan.digit_rows,
            },
        );
        stages.transform_native_rows(rows, I::INVERSE);
        publish_native(&&rows[..half], output, half, twiddles, I::INVERSE, scale);
    } else {
        let (data, work) = scratch.split_at_mut(WIDTH * half);
        input.stage(
            half,
            twiddles,
            down,
            &mut DigitMajor {
                values: data,
                order: &plan.digit_rows,
            },
        );
        stages.transform_staged_batch(data, work, I::INVERSE, S::ONE);
        publish_native(
            &LaneMajor {
                values: &*work,
                len: half,
            },
            output,
            half,
            twiddles,
            I::INVERSE,
            scale,
        );
    }
}

#[inline(always)]
fn publish_native<S: scioxide_core::RealScalar>(
    rows: &impl Load<Block<S, WIDTH>>,
    mut output: impl LaneWrite<Complex<S>>,
    half: usize,
    twiddles: &[Complex<S>],
    inverse: bool,
    scale: S,
) {
    if inverse {
        let eight = Reals::splat(S::TWO * S::TWO * S::TWO);
        for i in 0..half {
            let value = rows.load(i);
            output.write(
                2 * i,
                value.re.restore(eight).0.map(|v| Complex::new(v, S::ZERO)),
            );
            output.write(
                2 * i + 1,
                value.im.restore(eight).0.map(|v| Complex::new(v, S::ZERO)),
            );
        }
    } else {
        even::split_even(
            half,
            twiddles,
            rows,
            &BinScale(scale),
            &mut PackedOutput(output),
        );
    }
}

struct PackedOutput<V>(V);
impl<S: scioxide_core::RealScalar, V: LaneWrite<Complex<S>>> Store<Block<S, WIDTH>>
    for PackedOutput<V>
{
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        self.0
            .write(i, core::array::from_fn(|lane| value.lane(lane)));
    }
}

struct PartialLanes<V> {
    values: V,
    lanes: BlockLanes,
}
impl<T: Copy> LaneRead<T> for PartialLanes<&[T]> {
    #[inline(always)]
    fn read(&self, i: usize) -> [T; WIDTH] {
        self.lanes.read(self.values, i)
    }
}
impl<T: Copy> LaneWrite<T> for PartialLanes<&mut [T]> {
    #[inline(always)]
    fn write(&mut self, i: usize, lanes: [T; WIDTH]) {
        self.lanes.write(self.values, i, lanes);
    }
}

/// The even split of one block's half-length spectra into interleaved bins,
/// normalized by `scale` with positive zeros, as [`publish_even_forward`].
#[cfg(test)]
struct SplitRows<'p, 'r, S: scioxide_core::RealScalar> {
    plan: &'p RealFftPlan<S>,
    rows: &'r [Block<S, WIDTH>; ROWS],
    staging: &'r mut [Complex<S>],
    scale: S,
}

#[cfg(test)]
impl<S: scioxide_core::RealScalar> crate::kernel::block::BlockPass for SplitRows<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        if let Some((_, twiddles)) = self.plan.even_parts() {
            let half = self.plan.n / 2;
            even::split_even(
                half,
                twiddles,
                &&self.rows[..half],
                &BinScale(self.scale),
                &mut Interleaved(&mut self.staging[..WIDTH * (half + 1)]),
            );
        }
    }
}

/// Immutable normalization and zero canonicalization of a block.
struct BinScale<S>(S);
impl<S: scioxide_core::RealScalar> Scale<Block<S, WIDTH>> for BinScale<S> {
    #[inline(always)]
    fn scale(&self, value: Block<S, WIDTH>) -> Block<S, WIDTH> {
        let one = Reals::splat(S::ONE);
        Block::new(
            (value.re * self.0).restore(one),
            (value.im * self.0).restore(one),
        )
    }
}

/// The c2r merge of one block's interleaved bins, range-scaled by `2^-2`,
/// into digit-order rows, as [`preprocess_even_inverse`].
#[cfg(test)]
struct MergeRows<'p, 'r, S: scioxide_core::RealScalar> {
    plan: &'p RealFftPlan<S>,
    staging: &'r [Complex<S>],
    rows: &'r mut [Block<S, WIDTH>; ROWS],
}

#[cfg(test)]
impl<S: scioxide_core::RealScalar> crate::kernel::block::BlockPass for MergeRows<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        if let Some((_, twiddles)) = self.plan.even_parts() {
            even::merge_even(
                self.plan.n / 2,
                twiddles,
                &MergeBins(Interleaved(self.staging)),
                &mut &mut self.rows[..],
            );
        }
    }
}

/// Packed block input, range-scaled independently of its destination.
#[cfg(test)]
struct MergeBins<'a, S: scioxide_core::RealScalar>(Interleaved<&'a [Complex<S>]>);
#[cfg(test)]
impl<S: scioxide_core::RealScalar> Load<Block<S, WIDTH>> for MergeBins<'_, S> {
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, WIDTH> {
        let value: Block<S, WIDTH> = self.0.load(i);
        value * (S::ONE / (S::TWO + S::TWO))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::RealScalar;

    fn assert_bits<S: RealScalar>(actual: Complex<S>, expected: Complex<S>) {
        for (a, b) in [(actual.re, expected.re), (actual.im, expected.im)] {
            assert_eq!(
                a.try_to_f64_exact().unwrap().to_bits(),
                b.try_to_f64_exact().unwrap().to_bits()
            );
        }
    }

    fn native_real_bits<S: RealScalar>() {
        use crate::lanes::{LaneGeometry, Placement};
        let cx = ExecutionContext::sequential_native();
        for backend in crate::test_support::fft_backends(cx.processor()) {
            for n in [6, 8, 64, 256, 512] {
                for count in [1, 3, 4, 5] {
                    for axis in [0, 1] {
                        let m = n / 2 + 1;
                        let full = if axis == 0 { [n, count] } else { [count, n] };
                        let packed = if axis == 0 { [m, count] } else { [count, m] };
                        let fs = [full[1], 1];
                        let ps = [packed[1], 1];
                        let fg = LaneGeometry::new::<Complex<S>>(&full, &fs, axis);
                        let pg = LaneGeometry::new::<Complex<S>>(&packed, &ps, axis);
                        let input: Vec<_> = (0..n * count)
                            .map(|i| {
                                S::try_from_f64_lossy(((i * 17 % 97) as f64 - 48.0) / 128.0)
                                    .unwrap()
                            })
                            .collect();
                        let bins: Vec<_> = (0..m * count)
                            .map(|i| {
                                let k = if axis == 0 { i / count } else { i % m };
                                Complex::new(
                                    input[i],
                                    if k == 0 || k == n / 2 {
                                        -S::ZERO
                                    } else {
                                        input[i] * S::HALF
                                    },
                                )
                            })
                            .collect();
                        let plan = RealFftPlan::<S>::with_backend_and_context(
                            n,
                            ExecutionPolicy::Sequential,
                            backend,
                            cx,
                            scioxide_core::argument!("length"),
                            scioxide_core::argument!("context"),
                        )
                        .unwrap();
                        let mut alone = plan.clone();
                        alone.blocks = None;
                        for inverse in [false, true] {
                            let execute = |plan: &RealFftPlan<S>| {
                                let mut output = vec![
                                    Complex::new(S::ZERO, S::ZERO);
                                    if inverse { n * count } else { m * count }
                                ];
                                let mut staging = vec![
                                    Complex::new(S::ZERO, S::ZERO);
                                    fg.staging_len().max(pg.staging_len())
                                ];
                                let mut scratch =
                                    vec![Complex::new(S::ZERO, S::ZERO); plan.lane_scratch_len()];
                                let f = Placement::new(fg, &full, &fs);
                                let p = Placement::new(pg, &packed, &ps);
                                if inverse {
                                    plan.inverse_lanes(
                                        LanePairs::new((&bins, p), (&mut output, f), &mut staging),
                                        &mut scratch,
                                        LaneMode::Bounded,
                                        (S::ONE, S::ONE),
                                        cx,
                                    )
                                    .unwrap();
                                } else {
                                    plan.forward_lanes(
                                        LanePairs::new((&input, f), (&mut output, p), &mut staging),
                                        &mut scratch,
                                        LaneMode::Bounded,
                                        (BinaryScale::Identity, S::ONE),
                                        cx,
                                    )
                                    .unwrap();
                                }
                                output
                            };
                            for (a, b) in execute(&plan).into_iter().zip(execute(&alone)) {
                                assert_bits(a, b);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn native_blocks_equal_individual_real_lanes() {
        native_real_bits::<f32>();
        native_real_bits::<f64>();
    }

    fn split_merge_paths<S: RealScalar>() {
        let context = ExecutionContext::sequential_native();
        for backend in
            KernelBackend::available(context.processor()).filter(|b| *b != KernelBackend::Avx512)
        {
            for n in [2, 4, 6, 8, 10, 14, 18, 32, 64, 128, 256, 512] {
                let plan = RealFftPlan::<S>::with_backend_and_context(
                    n,
                    ExecutionPolicy::Sequential,
                    backend,
                    context,
                    scioxide_core::argument!("length"),
                    scioxide_core::argument!("context"),
                )
                .unwrap();
                let half = n / 2;
                let (_, twiddles) = plan.even_parts().unwrap();
                let dispatch = crate::backend::dispatch(backend, context.processor()).unwrap();
                let zero = Complex::new(S::ZERO, S::ZERO);
                let mut rows = [Block::from_fn(|_| zero); ROWS];
                for (i, row) in rows[..half].iter_mut().enumerate() {
                    *row = Block::from_fn(|lane| {
                        if lane == 0 {
                            return Complex::new(-S::ZERO, S::ZERO);
                        }
                        let re = ((i * 13 + lane * 7) % 31) as f64 / 32.0 - 0.5;
                        let im = ((i * 19 + lane * 11) % 29) as f64 / 32.0 - 0.5;
                        Complex::new(
                            S::try_from_f64_lossy(re).unwrap(),
                            S::try_from_f64_lossy(im).unwrap(),
                        )
                    });
                }
                let mut bins = vec![zero; WIDTH * (half + 1)];
                for scale in [S::ONE, S::HALF] {
                    crate::kernel::run_pass(
                        dispatch,
                        SplitRows {
                            plan: &plan,
                            rows: &rows,
                            staging: &mut bins,
                            scale,
                        },
                    );
                    for lane in 0..WIDTH {
                        let input: Vec<_> = rows[..half].iter().map(|row| row.lane(lane)).collect();
                        let mut expected = vec![zero; half + 1];
                        let publish = |value: Complex<S>| {
                            let value = value * scale;
                            Complex::new(
                                if value.re.is_zero() {
                                    S::ZERO
                                } else {
                                    value.re
                                },
                                if value.im.is_zero() {
                                    S::ZERO
                                } else {
                                    value.im
                                },
                            )
                        };
                        publish_even_forward(
                            &input,
                            twiddles,
                            &mut expected,
                            &crate::kernel::block::Map(|value: Block<S, 1>| {
                                Block::from(publish(value.lane(0)))
                            }),
                        );
                        let mut inplace = input;
                        inplace.push(zero);
                        postprocess_even_forward(&mut inplace, twiddles, half);
                        for (k, expected) in expected.into_iter().enumerate() {
                            assert_bits(bins[k * WIDTH + lane], expected);
                            assert_bits(publish(inplace[k]), expected);
                        }
                    }
                    let mut merged = [Block::from_fn(|_| zero); ROWS];
                    crate::kernel::run_pass(
                        dispatch,
                        MergeRows {
                            plan: &plan,
                            staging: &bins,
                            rows: &mut merged,
                        },
                    );
                    for lane in 0..WIDTH {
                        let mut scalar: Vec<_> =
                            bins.chunks_exact(WIDTH).map(|bin| bin[lane]).collect();
                        preprocess_even_inverse(&mut scalar, twiddles, half, BinaryScale::new(-2));
                        for (k, expected) in scalar[..half].iter().enumerate() {
                            assert_bits(merged[k].lane(lane), *expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_and_block_split_merge_agree_bit_for_bit_f32() {
        split_merge_paths::<f32>();
    }
    #[test]
    fn scalar_and_block_split_merge_agree_bit_for_bit_f64() {
        split_merge_paths::<f64>();
    }
}
