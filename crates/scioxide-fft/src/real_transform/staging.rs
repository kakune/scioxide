//! Mapped and scaled trigonometric input staging.
use super::*;

impl<S: scioxide_core::RealScalar> RealTransformPlan<S> {
    /// The half-length complex child of an even II/III/IV kernel, and whether
    /// it runs inverse (raw III).
    pub(super) fn even_child(&self) -> (&crate::complex::ComplexCore<S>, bool) {
        match &self.kernel {
            Kernel::EvenIv(child) => (child, false),
            Kernel::EvenHalf(real) => (
                real.even_parts()
                    .unwrap_or_else(|| unreachable!("even half kernel"))
                    .0,
                self.raw_type() == KernelType::Three,
            ),
            Kernel::Real(_) => unreachable!("only even kernels have a half-length child"),
        }
    }

    /// Stage one lane of an even II/III/IV kernel into `data` with the binary
    /// arm of `down` selected once, then run the unscaled half-length child.
    pub(super) fn even_spectrum<'a>(
        &self,
        x: impl Fn(usize) -> S,
        down: BinaryScale<S>,
        data: &'a mut [Complex<S>],
        work: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<&'a [Complex<S>], SciOxideError> {
        let x = Get(|i| Reals([x(i)]));
        let mut store = Put(|slot: usize, value: Block<S, 1>| data[slot] = value.lane(0));
        match down {
            BinaryScale::Identity => self.stage_even(&x, &Map(|value| value), &mut store),
            BinaryScale::Multiply(factor) => {
                self.stage_even(&x, &Map(|value| value * factor), &mut store);
            }
            BinaryScale::Extended(exponent) => self.stage_even(
                &x,
                &Map(|value: Reals<S, 1>| Reals([value.0[0].scalbn(exponent)])),
                &mut store,
            ),
        }
        let (child, inverse) = self.even_child();
        child.execute_bounded(data, work, inverse, S::ONE, context)?;
        Ok(data)
    }

    /// Stage mapped, range-scaled samples `down(x(i))` of an even kernel for
    /// its half-length child: Makhoul order for II, phase rotation fused with
    /// c2r combination for III, and rotated opposite-parity pairs for IV.
    /// See THIRD_PARTY_NOTICES.md for the IV packing.
    #[inline(always)]
    pub(super) fn stage_even<const L: usize>(
        &self,
        x: &impl Load<Reals<S, L>>,
        down: &impl Scale<Reals<S, L>>,
        store: &mut impl Store<Block<S, L>>,
    ) {
        let n = self.spec.length;
        let sine = matches!(self.spec.kind, RealTransformKind::Dst(_));
        match &self.kernel {
            Kernel::EvenIv(_) => {
                // The rotated real and negative imaginary parts are the even
                // and reversed odd cosine outputs; the sine map reverses the
                // input, swapping the packed components.
                for (i, &phase) in self.phases.iter().enumerate() {
                    let (even, odd) = (x.load(2 * i), x.load(n - 1 - 2 * i));
                    let (a, b) = if sine { (odd, even) } else { (even, odd) };
                    store.store(i, Block::new(down.scale(a), down.scale(b)) * phase);
                }
            }
            Kernel::EvenHalf(real) if self.raw_type() == KernelType::Three => {
                let (_, twiddles) = real
                    .even_parts()
                    .unwrap_or_else(|| unreachable!("even half kernel"));
                let samples = Samples { x, down, n, sine };
                stage_even_three(
                    store,
                    &self.phases,
                    twiddles,
                    self.scaling.input_first,
                    &samples,
                );
            }
            Kernel::EvenHalf(_) => {
                stage_even_two(
                    &mut ScaledPairs {
                        down,
                        output: store,
                    },
                    n,
                    x,
                    if sine { -S::ONE } else { S::ONE },
                );
            }
            Kernel::Real(_) => unreachable!("only even kernels stage half-length lanes"),
        }
    }

    /// Stage the range-scaled, sine-mapped, permuted and weighted input for
    /// the raw kernel in one loop, then run the unscaled child. Returns the
    /// child spectrum (the rotated half-length transform for even IV).
    pub(super) fn raw_spectrum<'a>(
        &self,
        input: &[S],
        work: &'a mut [Complex<S>],
        down: BinaryScale<S>,
        context: ExecutionContext<'_>,
    ) -> Result<&'a [Complex<S>], SciOxideError> {
        let n = self.spec.length;
        let sine = matches!(self.spec.kind, RealTransformKind::Dst(_));
        let phases = &self.phases;
        let scaling = self.scaling;
        let real = match &self.kernel {
            // Pack opposite-parity samples, rotate, and take an n/2-point DFT.
            Kernel::EvenIv(_) => {
                let (data, child) = work.split_at_mut(n / 2);
                return self.even_spectrum(|i| input[i], down, data, child, context);
            }
            Kernel::EvenHalf(real) => {
                let (data, child) = real.split_staging(work);
                return self.even_spectrum(|i| input[i], down, &mut data[..n / 2], child, context);
            }
            Kernel::Real(plan) => plan,
        };
        match self.raw_type() {
            KernelType::CosineOne => real.forward_bounded(
                work,
                |data| {
                    let extension = input.iter().chain(input[1..n - 1].iter().rev());
                    real.stage_samples(data, extension.copied(), down);
                    let first = staged_sample(real, data, 0);
                    *first *= scaling.input_first;
                    let last = staged_sample(real, data, n - 1);
                    *last *= scaling.input_last;
                },
                context,
            ),
            KernelType::SineOne => real.forward_bounded(
                work,
                |data| {
                    // Odd extension 0, x, 0, -reverse(x) of period 2(n+1).
                    let extension = core::iter::once(S::ZERO)
                        .chain(input.iter().copied())
                        .chain(core::iter::once(S::ZERO))
                        .chain(input.iter().rev().map(|&v| -v));
                    real.stage_samples(data, extension, down);
                },
                context,
            ),
            KernelType::Two => real.forward_bounded(
                work,
                |data| {
                    let sign = if sine { -S::ONE } else { S::ONE };
                    let evens = n.div_ceil(2);
                    real.stage_run(data, 0, evens, |j| down.apply(input[2 * j]));
                    real.stage_run(data, evens, n / 2, |j| {
                        down.apply(input[2 * (n / 2) - 1 - 2 * j] * sign)
                    });
                },
                context,
            ),
            KernelType::Three => real.forward_bounded(
                work,
                |data| {
                    debug_assert!(scaling.input_last == S::ONE);
                    stage_three(real, data, phases, scaling.input_first, input, down, sine);
                },
                context,
            ),
            KernelType::Four => real.forward_bounded(
                work,
                |data| {
                    if sine {
                        stage_odd_four(real, data, n, |i| down.apply(input[n - 1 - i]));
                    } else {
                        stage_odd_four(real, data, n, |i| down.apply(input[i]));
                    }
                },
                context,
            ),
        }
    }
}
/// Odd IV's apply_re11 walk has at most five contiguous strided runs.
/// Each run folds one reflected copy without a per-sample four-way branch.
/// The reduction and attribution remain documented in THIRD_PARTY_NOTICES.md.
fn stage_odd_four<S: scioxide_core::RealScalar>(
    real: &RealFftPlan<S>,
    data: &mut [Complex<S>],
    n: usize,
    sample: impl Fn(usize) -> S,
) {
    let mut first = 0;
    let mut m = n / 2;
    while first < n {
        let quadrant = m / n;
        let count = (((quadrant + 1) * n - m).div_ceil(4)).min(n - first);
        match quadrant {
            0 => real.stage_run(data, first, count, |j| sample(m + 4 * j)),
            1 => real.stage_run(data, first, count, |j| -sample(2 * n - 1 - m - 4 * j)),
            2 => real.stage_run(data, first, count, |j| -sample(m + 4 * j - 2 * n)),
            _ => real.stage_run(data, first, count, |j| sample(4 * n - 1 - m - 4 * j)),
        }
        first += count;
        m += 4 * count;
        if m >= 4 * n {
            m -= 4 * n;
        }
    }
}

/// Makhoul order, packed as even samples followed by reversed odd samples;
/// `x(i)` is sample `i`, mapped by `sign`. The destination applies range
/// scaling, optionally measuring the unscaled pairs during that same pass.
#[inline(always)]
pub(super) fn stage_even_two<S: scioxide_core::RealScalar, const L: usize>(
    store: &mut impl Store<Block<S, L>>,
    n: usize,
    x: &impl Load<Reals<S, L>>,
    sign: S,
) {
    let m = n / 2;
    for t in 0..n / 4 {
        let i = 4 * t;
        store.store(t, Block::new(x.load(i), x.load(i + 2)));
        store.store(
            m - 1 - t,
            Block::new(x.load(i + 3) * sign, x.load(i + 1) * sign),
        );
    }
    if !n.is_multiple_of(4) {
        store.store(n / 4, Block::new(x.load(n - 2), x.load(n - 1) * sign));
    }
}

/// Apply the binary operation after the II permutation and sine sign map.
struct ScaledPairs<'a, D, P> {
    down: &'a D,
    output: &'a mut P,
}
impl<S: scioxide_core::RealScalar, const L: usize, D: Scale<Reals<S, L>>, P: Store<Block<S, L>>>
    Store<Block<S, L>> for ScaledPairs<'_, D, P>
{
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, L>) {
        self.output.store(
            i,
            Block::new(self.down.scale(value.re), self.down.scale(value.im)),
        );
    }
}

/// Transpose of the even II reduction: fuse its inverse phase rotation with
/// c2r combination, reading four sample streams per mirror pair.
#[inline(always)]
fn stage_even_three<S: scioxide_core::RealScalar, const L: usize>(
    store: &mut impl Store<Block<S, L>>,
    phases: &[Complex<S>],
    twiddles: &[Complex<S>],
    first: S,
    sample: &impl Load<Reals<S, L>>,
) {
    crate::real_fft::merge_even(
        phases.len() - 1,
        twiddles,
        &EvenThree {
            sample,
            phases,
            first,
        },
        store,
    );
}

/// Spectrum bins of III's transposed II reduction. Endpoints have their
/// own factors; interior bins keep the inverse phase rotation's order.
struct EvenThree<'a, S, X> {
    sample: &'a X,
    phases: &'a [Complex<S>],
    first: S,
}

impl<S: scioxide_core::RealScalar, const L: usize, X: Load<Reals<S, L>>> Load<Block<S, L>>
    for EvenThree<'_, S, X>
{
    #[inline(always)]
    fn load(&self, k: usize) -> Block<S, L> {
        bin_three(self.sample, self.phases, k)
    }

    #[inline(always)]
    fn load_endpoint(&self, k: usize) -> Block<S, L> {
        if k == 0 {
            Block::real(self.sample.load(0) * self.first)
        } else {
            Block::real(self.sample.load(k) * (S::TWO * self.phases[k].re))
        }
    }
}

/// Inverse phase rotation of the III input bin `j`: `conj(phase[j]) *
/// (x[j] - i x[n - j])`.
#[inline(always)]
fn bin_three<S: scioxide_core::RealScalar, const L: usize>(
    sample: &impl Load<Reals<S, L>>,
    phases: &[Complex<S>],
    j: usize,
) -> Block<S, L> {
    let n = 2 * (phases.len() - 1);
    Block::new(sample.load(j), -sample.load(n - j)) * phases[j].conj()
}

/// The mapped, range-scaled III samples `down(x(i))`, reversed for sine.
struct Samples<'a, X, D> {
    x: &'a X,
    down: &'a D,
    n: usize,
    sine: bool,
}

impl<S: scioxide_core::RealScalar, const L: usize, X, D> Load<Reals<S, L>> for Samples<'_, X, D>
where
    X: Load<Reals<S, L>>,
    D: Scale<Reals<S, L>>,
{
    #[inline(always)]
    fn load(&self, i: usize) -> Reals<S, L> {
        self.down
            .scale(self.x.load(if self.sine { self.n - 1 - i } else { i }))
    }
}

/// Stage odd DCT-III's transposed real FFT from mapped, range-scaled input.
/// Select the scale before the paired loops so their small callbacks inline.
fn stage_three<S: scioxide_core::RealScalar>(
    real: &RealFftPlan<S>,
    data: &mut [Complex<S>],
    phases: &[Complex<S>],
    first: S,
    input: &[S],
    down: BinaryScale<S>,
    sine: bool,
) {
    match down {
        BinaryScale::Identity => {
            stage_three_scaled(real, data, phases, (first, sine), input, |v| v)
        }
        BinaryScale::Multiply(factor) => {
            stage_three_scaled(real, data, phases, (first, sine), input, |v| v * factor)
        }
        BinaryScale::Extended(exponent) => {
            stage_three_scaled(real, data, phases, (first, sine), input, |v| {
                v.scalbn(exponent)
            })
        }
    }
}

fn stage_three_scaled<S: scioxide_core::RealScalar>(
    real: &RealFftPlan<S>,
    data: &mut [Complex<S>],
    phases: &[Complex<S>],
    (first, sine): (S, bool),
    input: &[S],
    down: impl Fn(S) -> S,
) {
    let n = real.size();
    debug_assert!(!n.is_multiple_of(2));
    let half = n / 2;
    // Expose the common child-derived extent of all three streams so the
    // compiler can eliminate per-pair input and phase length checks.
    let phases = &phases[1..=half];
    let pair = |phase: Complex<S>, a: S, b: S| {
        let (a, b) = (down(a), down(b));
        let (sum, difference) = (a + b, a - b);
        (
            phase.re * difference - phase.im * sum,
            phase.re * sum + phase.im * difference,
        )
    };
    if sine {
        let low = &input[half..n - 1];
        let high = &input[..half];
        stage_pairs(real, data, down(input[n - 1]) * first, |k| {
            pair(phases[k - 1], low[half - k], high[k - 1])
        });
    } else {
        let low = &input[1..=half];
        let high = &input[half + 1..n];
        stage_pairs(real, data, down(input[0]) * first, |k| {
            pair(phases[k - 1], low[k - 1], high[half - k])
        });
    }
}

/// Access a staged endpoint to apply trigonometric input weights.
fn staged_sample<'a, S: scioxide_core::RealScalar>(
    real: &RealFftPlan<S>,
    data: &'a mut [Complex<S>],
    index: usize,
) -> &'a mut S {
    if !real.is_packed() {
        &mut data[index].re
    } else if index.is_multiple_of(2) {
        &mut data[index / 2].re
    } else {
        &mut data[index / 2].im
    }
}

/// Stage an odd III reduction: `first` is sample zero and `pair(k)`
/// produces samples `k` and `n-k` for `1 <= k <= n/2`. Their opposite
/// parities fill packed components in two branch-free loops.
#[inline]
fn stage_pairs<S: scioxide_core::RealScalar>(
    real: &RealFftPlan<S>,
    data: &mut [Complex<S>],
    first: S,
    pair: impl Fn(usize) -> (S, S),
) {
    debug_assert!(real.size() > 2 && !real.size().is_multiple_of(2));
    // Make the full staging extent visible to the loop's bounds proofs.
    let data = &mut data[..if real.is_packed() {
        real.size().div_ceil(2)
    } else {
        real.size()
    }];
    let half = real.size() / 2;
    if real.is_packed() {
        // Odd n gives k and n - k opposite parity.
        data[0].re = first;
        for k in (1..=half).step_by(2) {
            let (a, b) = pair(k);
            data[k / 2].im = a;
            data[(real.size() - k) / 2].re = b;
        }
        for k in (2..=half).step_by(2) {
            let (a, b) = pair(k);
            data[k / 2].re = a;
            data[(real.size() - k) / 2].im = b;
        }
    } else {
        data[0] = Complex::new(first, S::ZERO);
        for k in 1..=half {
            let (a, b) = pair(k);
            data[k] = Complex::new(a, S::ZERO);
            data[real.size() - k] = Complex::new(b, S::ZERO);
        }
    }
}
