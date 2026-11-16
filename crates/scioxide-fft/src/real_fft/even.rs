//! Shared even-length split and merge arithmetic.
use super::*;
use core::cell::Cell;

/// Even-length split of `Z = FFT_m(z)`, `z[j] = x[2j] + i x[2j+1]`, into the
/// interior bin `k` of `X`, from `left = Z[k]` and `right = Z[m-k]`.
///
/// The even/odd sample spectra are `E[k] = (Z[k] + conj(Z[m-k])) / 2` and
/// `O[k] = -i (Z[k] - conj(Z[m-k])) / 2`, and `X[k] = E[k] + w^k O[k]`.
#[inline(always)]
fn split<S: scioxide_core::RealScalar, const L: usize>(
    left: Block<S, L>,
    right: Block<S, L>,
    twiddle: Complex<S>,
) -> Block<S, L> {
    let imaginary = Complex::new(S::ZERO, S::ONE);
    ((left + right.conj()) - (left - right.conj()) * (imaginary * twiddle)) * S::HALF
}

/// The c2r pre-step of an even real inverse, the transpose of [`split`]:
/// `((L + conj(R)) + i conj(w) (L - conj(R))) / 2`.
#[inline(always)]
fn merge<S: scioxide_core::RealScalar, const L: usize>(
    left: Block<S, L>,
    right: Block<S, L>,
    twiddle: Complex<S>,
) -> Block<S, L> {
    let imaginary = Complex::new(S::ZERO, S::ONE);
    ((left + right.conj()) + (left - right.conj()) * (imaginary * twiddle.conj())) * S::HALF
}

/// Split DC, Nyquist and mirror pairs through one pair-safe accessor.
/// Reads of a mirror pair precede either write, permitting in-place storage.
#[inline(always)]
pub(crate) fn split_even<S: scioxide_core::RealScalar, const L: usize>(
    half: usize,
    twiddles: &[Complex<S>],
    raw: &impl Load<Block<S, L>>,
    publish: &impl Scale<Block<S, L>>,
    output: &mut impl Store<Block<S, L>>,
) {
    let twiddles = &twiddles[..=half];
    let z = raw.load(0);
    let zero = Reals::splat(S::ZERO);
    output.store_endpoint(0, publish.scale(Block::new(z.re + z.im, zero)));
    output.store_endpoint(half, publish.scale(Block::new(z.re - z.im, zero)));
    for k in 1..half.div_ceil(2) {
        let mirror = half - k;
        let (left, right) = (raw.load(k), raw.load(mirror));
        output.store(k, publish.scale(split(left, right, twiddles[k])));
        output.store(mirror, publish.scale(split(right, left, twiddles[mirror])));
    }
    if half.is_multiple_of(2) {
        let k = half / 2;
        let value = raw.load(k);
        output.store(k, publish.scale(split(value, value, twiddles[k])));
    }
}

/// Merge endpoint and mirror pairs with the same arithmetic for every lane.
#[inline(always)]
pub(crate) fn merge_even<S: scioxide_core::RealScalar, const L: usize>(
    half: usize,
    twiddles: &[Complex<S>],
    raw: &impl Load<Block<S, L>>,
    output: &mut impl Store<Block<S, L>>,
) {
    let twiddles = &twiddles[..=half];
    output.store(
        0,
        merge(raw.load_endpoint(0), raw.load_endpoint(half), twiddles[0]),
    );
    for k in 1..half.div_ceil(2) {
        let mirror = half - k;
        let (left, right) = (raw.load(k), raw.load(mirror));
        output.store(k, merge(left, right, twiddles[k]));
        output.store(mirror, merge(right, left, twiddles[mirror]));
    }
    if half.is_multiple_of(2) {
        let k = half / 2;
        let value = raw.load(k);
        output.store(k, merge(value, value, twiddles[k]));
    }
}

/// Aliased scalar staging through safe cells. The shared kernels read both
/// mirror inputs before replacing either, including when these views overlap.
#[derive(Clone, Copy)]
struct ScalarSlots<'a, S> {
    values: &'a [Cell<Complex<S>>],
    down: BinaryScale<S>,
}
impl<S: scioxide_core::RealScalar> Load<Block<S, 1>> for ScalarSlots<'_, S> {
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, 1> {
        let value = self.values[i].get();
        Block::from(Complex::new(
            self.down.apply(value.re),
            self.down.apply(value.im),
        ))
    }
}
impl<S: scioxide_core::RealScalar> Store<Block<S, 1>> for ScalarSlots<'_, S> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, 1>) {
        self.values[i].set(value.lane(0));
    }
}

/// In-place even split, including its extra Nyquist slot.
pub(super) fn postprocess_even_forward<S: scioxide_core::RealScalar>(
    data: &mut [Complex<S>],
    twiddles: &[Complex<S>],
    half: usize,
) {
    let raw = ScalarSlots {
        values: Cell::from_mut(&mut data[..=half]).as_slice_of_cells(),
        down: BinaryScale::Identity,
    };
    let mut output = raw;
    split_even(
        half,
        twiddles,
        &raw,
        &crate::kernel::block::Map(|value| value),
        &mut output,
    );
}

/// In-place range-scaled even inverse pre-step.
pub(super) fn preprocess_even_inverse<S: scioxide_core::RealScalar>(
    data: &mut [Complex<S>],
    twiddles: &[Complex<S>],
    half: usize,
    down: BinaryScale<S>,
) {
    let raw = ScalarSlots {
        values: Cell::from_mut(&mut data[..=half]).as_slice_of_cells(),
        down,
    };
    let mut output = raw;
    merge_even(half, twiddles, &raw, &mut output);
}
