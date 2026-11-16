//! Trigonometric phase rotation and output publication.
use super::*;

impl<S: scioxide_core::RealScalar> RealTransformPlan<S> {
    /// Select binary restoration once for the entire publication pass.
    pub(super) fn publish_restored(
        &self,
        raw: &[Complex<S>],
        restore: BinaryScale<S>,
        mut put: impl FnMut(usize, S),
    ) {
        let raw = |k: usize| Block::from(raw[k]);
        let canonical = |value: S, scaled: S| if value.is_zero() { S::ZERO } else { scaled };
        match restore {
            BinaryScale::Identity => self.publish(raw, |i, value| {
                let value = value.0[0];
                put(i, canonical(value, value));
            }),
            BinaryScale::Multiply(factor) => self.publish(raw, |i, value| {
                let value = value.0[0];
                put(i, canonical(value, value * factor));
            }),
            BinaryScale::Extended(exponent) => self.publish(raw, |i, value| {
                let value = value.0[0];
                put(i, canonical(value, value.scalbn(exponent)));
            }),
        }
    }

    /// Post-process the raw spectrum `raw(k)` of `L` lanes and pass each
    /// normalized output `(index, values)` to `put` before restoration. The
    /// sine reversal is selected once; uniform interior factors stay in the
    /// loops while endpoint factors and the sine `(-1)^k` sign are folded into
    /// the fixed statements that produce them.
    pub(super) fn publish(
        &self,
        raw: impl Fn(usize) -> Block<S, 1>,
        mut put: impl FnMut(usize, Reals<S, 1>),
    ) {
        let n = self.spec.length;
        match self.sine_map() {
            (true, _) => self.post(raw, false, |i, value| put(n - 1 - i, value)),
            (false, alternate) => self.post(raw, alternate, put),
        }
    }

    /// Lane form of the single-lane output pass, for an even II/III/IV kernel
    /// over `L` lanes.
    #[inline(always)]
    pub(super) fn publish_even<const L: usize>(
        &self,
        raw: &impl Load<Block<S, L>>,
        put: &mut impl Store<Reals<S, L>>,
    ) {
        let last = self.spec.length - 1;
        match self.sine_map() {
            (true, _) => self.post_even(raw, false, &mut Reversed { put, last }),
            (false, alternate) => self.post_even(raw, alternate, put),
        }
    }

    /// Sine output maps over the cosine kernels: raw II reverses the outputs
    /// and raw III/IV alternate the signs of odd outputs.
    pub(super) fn sine_map(&self) -> (bool, bool) {
        let sine = matches!(self.spec.kind, RealTransformKind::Dst(_));
        match self.raw_type() {
            KernelType::Two => (sine, false),
            KernelType::Three | KernelType::Four => (false, sine),
            KernelType::CosineOne | KernelType::SineOne => (false, false),
        }
    }

    pub(super) fn post(
        &self,
        raw: impl Fn(usize) -> Block<S, 1>,
        alternate: bool,
        mut put: impl FnMut(usize, Reals<S, 1>),
    ) {
        let n = self.spec.length;
        let phases = &self.phases;
        let scaling = self.scaling;
        let (f, first, last) = (
            scaling.output_scale,
            scaling.output_first,
            scaling.output_last,
        );
        // Factors at odd output indices carry the alternating sine sign.
        let odd = |factor: S| if alternate { -factor } else { factor };
        match (self.raw_type(), &self.kernel) {
            (_, Kernel::EvenHalf(_) | Kernel::EvenIv(_)) => {
                self.post_even(&Get(raw), alternate, &mut Put(put));
            }
            (KernelType::CosineOne, _) => {
                put(0, raw(0).re * first);
                for k in 1..n - 1 {
                    put(k, raw(k).re * f);
                }
                put(n - 1, raw(n - 1).re * last);
            }
            (KernelType::SineOne, _) => {
                for k in 0..n {
                    put(k, (raw(k + 1).im * -S::HALF) * f);
                }
            }
            (KernelType::Two, _) => {
                let rotated = |k: usize| raw(k) * phases[k];
                put(0, rotated(0).re * first);
                let value = rotated(1);
                put(1, value.re * f);
                put(n - 1, -value.im * last);
                for k in 2..n.div_ceil(2) {
                    let value = rotated(k);
                    put(k, value.re * f);
                    put(n - k, -value.im * f);
                }
                if n.is_multiple_of(2) {
                    put(n / 2, rotated(n / 2).re * f);
                }
            }
            (KernelType::Three, _) => {
                put(0, raw(0).re * first);
                for k in 1..n / 2 {
                    let value = raw(k);
                    put(2 * k - 1, (value.re - value.im) * odd(f));
                    put(2 * k, (value.re + value.im) * f);
                }
                let value = raw(n / 2);
                if n.is_multiple_of(2) {
                    put(n - 1, (value.re - value.im) * odd(last));
                } else {
                    put(n - 2, (value.re - value.im) * odd(f));
                    put(n - 1, (value.re + value.im) * last);
                }
            }
            (KernelType::Four, Kernel::Real(_)) => {
                let half = n / 2;
                let sign = |i: usize| if i & 2 == 0 { S::ONE } else { -S::ONE };
                let factor = |i: usize| if i % 2 == 1 { odd(f) } else { f };
                put(half, raw(0).re * sign(half + 1) * factor(half));
                // Odd and even Fourier bins have different index maps. Each
                // residue modulo four fixes both output signs and factors.
                for residue in 0..4 {
                    let (a, b) = (residue, n.wrapping_sub(1 + residue));
                    let (ar, ai, br, bi) = (sign(a + 1), sign(a), sign(b.wrapping_add(1)), sign(b));
                    let (af, bf) = (factor(a), factor(b));
                    for i in (residue..half.div_ceil(2)).step_by(4) {
                        let value = raw(2 * i + 1);
                        put(i, (value.re * ar + value.im * ai) * af);
                        put(n - 1 - i, (value.re * br - value.im * bi) * bf);
                    }
                }
                for residue in 1..=4 {
                    let (a, b) = (half.wrapping_sub(residue), half + residue);
                    let (ar, ai, br, bi) = (sign(a.wrapping_add(1)), sign(a), sign(b + 1), sign(b));
                    let (af, bf) = (factor(a), factor(b));
                    for i in (residue..=half / 2).step_by(4) {
                        let value = raw(2 * i);
                        let im = -value.im;
                        put(half - i, (value.re * ar + im * ai) * af);
                        put(half + i, (value.re * br - im * bi) * bf);
                    }
                }
            }
        }
    }

    /// [`Self::post`] for the even II/III/IV kernels over `L` lanes.
    #[inline(always)]
    pub(super) fn post_even<const L: usize>(
        &self,
        raw: &impl Load<Block<S, L>>,
        alternate: bool,
        put: &mut impl Store<Reals<S, L>>,
    ) {
        let n = self.spec.length;
        let phases = &self.phases;
        let scaling = self.scaling;
        let (f, first) = (scaling.output_scale, scaling.output_first);
        // Factors at odd output indices carry the alternating sine sign.
        let odd = |factor: S| if alternate { -factor } else { factor };
        match (self.raw_type(), &self.kernel) {
            (KernelType::Two, Kernel::EvenHalf(real)) => {
                let (_, twiddles) = real
                    .even_parts()
                    .unwrap_or_else(|| unreachable!("even half kernel"));
                crate::real_fft::split_even(
                    n / 2,
                    twiddles,
                    raw,
                    &Map(|value| value),
                    &mut EvenTwo {
                        put,
                        phases,
                        first,
                        factor: f,
                    },
                );
            }
            (KernelType::Three, Kernel::EvenHalf(_)) => {
                // The inverse half-length FFT supplies half the raw III
                // result. Fold the exact factor two locally, leaving Scaling
                // and every candidate/direct publication path unchanged.
                let f = f + f;
                let odd = odd(f);
                let m = n / 2;
                for t in 0..n / 4 {
                    let (left, right) = (raw.load(t), raw.load(m - 1 - t));
                    put.store(4 * t, left.re * f);
                    put.store(4 * t + 2, left.im * f);
                    put.store(4 * t + 1, right.im * odd);
                    put.store(4 * t + 3, right.re * odd);
                }
                if !n.is_multiple_of(4) {
                    let value = raw.load(n / 4);
                    put.store(n - 2, value.re * f);
                    put.store(n - 1, value.im * odd);
                }
            }
            (KernelType::Four, Kernel::EvenIv(_)) => {
                for (i, &phase) in phases.iter().enumerate() {
                    let value = raw.load(i) * phase;
                    put.store(2 * i, value.re * f);
                    put.store(n - 1 - 2 * i, -value.im * odd(f));
                }
            }
            _ => unreachable!("only even kernels publish half-length spectra"),
        }
    }
}
/// Outputs reversed by the sine II map.
struct Reversed<'a, P> {
    put: &'a mut P,
    last: usize,
}

impl<T, P: Store<T>> Store<T> for Reversed<'_, P> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: T) {
        self.put.store(self.last - i, value);
    }
}

/// Rotate each split bin and write its cosine pair. DC and Nyquist each
/// produce only one output; interior bins map to `k` and `n - k`.
struct EvenTwo<'a, S, P> {
    put: &'a mut P,
    phases: &'a [Complex<S>],
    first: S,
    factor: S,
}

impl<S: scioxide_core::RealScalar, const L: usize, P: Store<Reals<S, L>>> Store<Block<S, L>>
    for EvenTwo<'_, S, P>
{
    #[inline(always)]
    fn store(&mut self, k: usize, value: Block<S, L>) {
        let value = value * self.phases[k];
        self.put.store(k, value.re * self.factor);
        self.put
            .store(2 * (self.phases.len() - 1) - k, -value.im * self.factor);
    }

    #[inline(always)]
    fn store_endpoint(&mut self, k: usize, value: Block<S, L>) {
        let value = value * self.phases[k];
        self.put
            .store(k, value.re * if k == 0 { self.first } else { self.factor });
    }
}
