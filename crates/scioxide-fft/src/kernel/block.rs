//! Values of several independent lanes at one position.
//!
//! A [`Block`] holds the real and imaginary components of `L` lanes as two
//! arrays, so one complex operation becomes element-wise operations across the
//! lanes and twiddles are shared scalars. Every operation evaluates, lane by
//! lane, exactly the expression of the matching [`Complex`] operation; Rust
//! never contracts separate multiplies and additions, so a lane's result does
//! not depend on `L`, on the other lanes or on the build's target features.
//! The explicit `rotate_fused` adapter instead mirrors AVX2's complex product
//! when a native row pass selects it; ordinary operators remain non-fused.
//! One-dimensional plans use `L = 1`; N-D lane passes use [`WIDTH`].
use super::scalar::Butterfly;
use core::ops::{Add, Mul, Neg, Sub};

use scioxide_core::{Complex, RealScalar};

/// Consecutive lanes `first + j * step` for `j < count`, each holding `len`
/// values `stride` apart.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Run {
    pub(crate) first: usize,
    pub(crate) step: usize,
    pub(crate) count: usize,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

/// Lanes processed together by N-D lane passes.
pub(crate) const WIDTH: usize = 4;

/// Longest child whose lane blocks a bounded pass keeps on the stack: one
/// block row per point, 16 KiB of `f64` rows at most.
pub(crate) const ROWS: usize = 256;

/// Longest complex lane run across lanes. Gathering complex lanes in digit
/// order costs more than staging real ones; beyond this length in-place
/// per-lane leaves are as fast for contiguous lanes, and the choice may
/// depend only on the length so that every layout takes one arithmetic.
pub(crate) const COMPLEX_ROWS: usize = 32;

/// One lane-block pass. [`super::run_pass`] runs its body as compiled for the
/// plan's verified backend, so implementations and every block function
/// they reach are `#[inline(always)]`; per-lane child calls stay calls.
pub(crate) trait BlockPass {
    type Output;
    fn run(self) -> Self::Output;
}

/// Indexed reads of lane values. Lane passes take accessors through these
/// traits, whose methods are `#[inline(always)]`, rather than closures,
/// which are not reliably inlined into a pass's wider build.
pub(crate) trait Load<T> {
    fn load(&self, i: usize) -> T;

    /// Read an endpoint, allowing transform adapters to apply endpoint weights
    /// without testing indices on interior reads. Ordinary storage reads alike.
    #[inline(always)]
    fn load_endpoint(&self, i: usize) -> T {
        self.load(i)
    }
}

/// Indexed writes of lane values; see [`Load`].
pub(crate) trait Store<T> {
    fn store(&mut self, i: usize, value: T);

    /// Write an endpoint separately from interior output mappings. Ordinary
    /// storage writes alike; transform adapters may specialize their factors.
    #[inline(always)]
    fn store_endpoint(&mut self, i: usize, value: T) {
        self.store(i, value);
    }
}

/// The range scaling applied to staged samples; see [`Load`].
pub(crate) trait Scale<T> {
    fn scale(&self, value: T) -> T;
}

/// One-lane closure reads.
pub(crate) struct Get<F>(pub(crate) F);
impl<T, F: Fn(usize) -> T> Load<T> for Get<F> {
    #[inline(always)]
    fn load(&self, i: usize) -> T {
        (self.0)(i)
    }
}

/// One-lane closure writes.
pub(crate) struct Put<F>(pub(crate) F);
impl<T, F: FnMut(usize, T)> Store<T> for Put<F> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: T) {
        (self.0)(i, value);
    }
}

/// One-lane closure scaling.
pub(crate) struct Map<F>(pub(crate) F);
impl<T, F: Fn(T) -> T> Scale<T> for Map<F> {
    #[inline(always)]
    fn scale(&self, value: T) -> T {
        (self.0)(value)
    }
}

/// Multiplication of each lane by its own ordinary binary factor.
impl<S: RealScalar, const L: usize> Scale<Reals<S, L>> for Reals<S, L> {
    #[inline(always)]
    fn scale(&self, value: Self) -> Self {
        value * *self
    }
}

/// Lanes `block..block + WIDTH` of a run, read in place at each position.
pub(crate) trait LaneRead<T> {
    fn read(&self, i: usize) -> [T; WIDTH];
}

/// Lanes `block..block + WIDTH` of a run, written in place at each position.
pub(crate) trait LaneWrite<T> {
    fn write(&mut self, i: usize, lanes: [T; WIDTH]);
}

/// How the lanes of a run lie in memory. A pass is compiled once for each
/// pattern: adjacent lanes (`step == 1`) move as one slice per position,
/// contiguous lanes (`stride == 1`) as equal-length rows sharing one bounds
/// check, and any other run element by element.
pub(crate) trait Addressing {
    type Read<'a, T: Copy + 'a>: LaneRead<T>;
    type Write<'a, T: Copy + 'a>: LaneWrite<T>;
    fn read<T: Copy>(values: &[T], run: Run, block: usize) -> Self::Read<'_, T>;
    fn write<T: Copy>(values: &mut [T], run: Run, block: usize) -> Self::Write<'_, T>;
}

/// Lanes one element apart.
pub(crate) enum Adjacent {}
/// Lanes whose elements are adjacent.
pub(crate) enum Contiguous {}
/// Any other run.
pub(crate) enum Spread {}

/// The addressing of `run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Adjacent,
    Contiguous,
    Spread,
}

impl Layout {
    pub(crate) const fn of(run: Run) -> Self {
        if run.step == 1 {
            Self::Adjacent
        } else if run.stride == 1 && run.step >= run.len {
            Self::Contiguous
        } else {
            Self::Spread
        }
    }
}

/// Lanes at `first + lane + i * stride`.
#[derive(Debug)]
pub(crate) struct AdjacentLanes<V> {
    values: V,
    first: usize,
    stride: usize,
}

impl<T: Copy> LaneRead<T> for AdjacentLanes<&[T]> {
    #[inline(always)]
    fn read(&self, i: usize) -> [T; WIDTH] {
        let at = &self.values[self.first + i * self.stride..][..WIDTH];
        core::array::from_fn(|lane| at[lane])
    }
}

impl<T: Copy> LaneWrite<T> for AdjacentLanes<&mut [T]> {
    #[inline(always)]
    fn write(&mut self, i: usize, lanes: [T; WIDTH]) {
        self.values[self.first + i * self.stride..][..WIDTH].copy_from_slice(&lanes);
    }
}

impl Addressing for Adjacent {
    type Read<'a, T: Copy + 'a> = AdjacentLanes<&'a [T]>;
    type Write<'a, T: Copy + 'a> = AdjacentLanes<&'a mut [T]>;
    #[inline(always)]
    fn read<T: Copy>(values: &[T], run: Run, block: usize) -> Self::Read<'_, T> {
        AdjacentLanes {
            values,
            first: run.first + block,
            stride: run.stride,
        }
    }
    #[inline(always)]
    fn write<T: Copy>(values: &mut [T], run: Run, block: usize) -> Self::Write<'_, T> {
        AdjacentLanes {
            values,
            first: run.first + block,
            stride: run.stride,
        }
    }
}

/// Lanes as rows of `run.len` adjacent elements.
#[derive(Debug)]
pub(crate) struct LaneRows<V>([V; WIDTH]);

impl<T: Copy> LaneRead<T> for LaneRows<&[T]> {
    #[inline(always)]
    fn read(&self, i: usize) -> [T; WIDTH] {
        core::array::from_fn(|lane| self.0[lane][i])
    }
}

impl<T: Copy> LaneWrite<T> for LaneRows<&mut [T]> {
    #[inline(always)]
    fn write(&mut self, i: usize, lanes: [T; WIDTH]) {
        for (row, value) in self.0.iter_mut().zip(lanes) {
            row[i] = value;
        }
    }
}

impl Addressing for Contiguous {
    type Read<'a, T: Copy + 'a> = LaneRows<&'a [T]>;
    type Write<'a, T: Copy + 'a> = LaneRows<&'a mut [T]>;
    #[inline(always)]
    fn read<T: Copy>(values: &[T], run: Run, block: usize) -> Self::Read<'_, T> {
        let first = run.first + block * run.step;
        LaneRows(core::array::from_fn(|lane| {
            &values[first + lane * run.step..][..run.len]
        }))
    }
    #[inline(always)]
    fn write<T: Copy>(values: &mut [T], run: Run, block: usize) -> Self::Write<'_, T> {
        // Rows are `step >= len` apart, so each split keeps one whole row.
        let mut rest = &mut values[run.first + block * run.step..];
        LaneRows(core::array::from_fn(|_| {
            let rows = core::mem::take(&mut rest);
            let (row, tail) = rows.split_at_mut(run.step.min(rows.len()));
            rest = tail;
            &mut row[..run.len]
        }))
    }
}

/// Lanes at `first + lane * step + i * stride`.
#[derive(Debug)]
pub(crate) struct SpreadLanes<V> {
    values: V,
    first: usize,
    step: usize,
    stride: usize,
}

impl<T: Copy> LaneRead<T> for SpreadLanes<&[T]> {
    #[inline(always)]
    fn read(&self, i: usize) -> [T; WIDTH] {
        let start = self.first + i * self.stride;
        core::array::from_fn(|lane| self.values[start + lane * self.step])
    }
}

impl<T: Copy> LaneWrite<T> for SpreadLanes<&mut [T]> {
    #[inline(always)]
    fn write(&mut self, i: usize, lanes: [T; WIDTH]) {
        let start = self.first + i * self.stride;
        for (lane, value) in lanes.into_iter().enumerate() {
            self.values[start + lane * self.step] = value;
        }
    }
}

impl Addressing for Spread {
    type Read<'a, T: Copy + 'a> = SpreadLanes<&'a [T]>;
    type Write<'a, T: Copy + 'a> = SpreadLanes<&'a mut [T]>;
    #[inline(always)]
    fn read<T: Copy>(values: &[T], run: Run, block: usize) -> Self::Read<'_, T> {
        SpreadLanes {
            values,
            first: run.first + block * run.step,
            step: run.step,
            stride: run.stride,
        }
    }
    #[inline(always)]
    fn write<T: Copy>(values: &mut [T], run: Run, block: usize) -> Self::Write<'_, T> {
        SpreadLanes {
            values,
            first: run.first + block * run.step,
            step: run.step,
            stride: run.stride,
        }
    }
}

/// Real lane values interleaved by position: lane `l` of element `i` at
/// `i * WIDTH + l`. Staging and post-processing then load and store every
/// position's lanes as one slice, whatever the lanes' own layout.
#[derive(Debug)]
pub(crate) struct Interleaved<V>(pub(crate) V);

/// Block access to consecutive natural-order complex lanes.
pub(crate) struct LaneMajor<V> {
    pub(crate) values: V,
    pub(crate) len: usize,
}

/// Bounded block outputs placed directly in each native lane's digit order.
pub(crate) struct DigitMajor<'a, S> {
    pub(crate) values: &'a mut [Complex<S>],
    pub(crate) order: &'a [usize],
}

/// Block values staged directly in a native child's digit order.
pub(crate) struct PermutedRows<'a, S: RealScalar> {
    pub(crate) rows: &'a mut [Block<S, WIDTH>],
    pub(crate) order: &'a [usize],
}
impl<S: RealScalar> Store<Block<S, WIDTH>> for PermutedRows<'_, S> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        self.rows[self.order[i]] = value;
    }
}

impl<S: RealScalar> Store<Block<S, WIDTH>> for DigitMajor<'_, S> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        let index = self.order[i];
        let value = value.bounded();
        for lane in 0..WIDTH {
            self.values[lane * self.order.len() + index] = value.lane(lane);
        }
    }
}

impl<S: RealScalar> Load<Block<S, WIDTH>> for LaneMajor<&[Complex<S>]> {
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, WIDTH> {
        Block::from_fn(|lane| self.values[lane * self.len + i])
    }
}

impl<S: RealScalar> Store<Block<S, WIDTH>> for LaneMajor<&mut [Complex<S>]> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        for (lane, row) in self.values.chunks_exact_mut(self.len).enumerate() {
            row[i] = value.lane(lane);
        }
    }
}

/// Block reads and writes using a run's prepared addressing.
pub(crate) struct RealLanes<V>(pub(crate) V);

impl<S: RealScalar, V: LaneRead<S>> Load<Reals<S, WIDTH>> for RealLanes<V> {
    #[inline(always)]
    fn load(&self, i: usize) -> Reals<S, WIDTH> {
        Reals(self.0.read(i))
    }
}

impl<S: RealScalar, V: LaneWrite<S>> Store<Reals<S, WIDTH>> for RealLanes<V> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Reals<S, WIDTH>) {
        self.0.write(i, value.0);
    }
}

impl<S: RealScalar> Load<Reals<S, WIDTH>> for Interleaved<&[S]> {
    #[inline(always)]
    fn load(&self, i: usize) -> Reals<S, WIDTH> {
        let at = &self.0[i * WIDTH..][..WIDTH];
        Reals::from_fn(|lane| at[lane])
    }
}

impl<S: RealScalar> Store<Reals<S, WIDTH>> for Interleaved<&mut [S]> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Reals<S, WIDTH>) {
        self.0[i * WIDTH..][..WIDTH].copy_from_slice(&value.0);
    }
}

impl<S: RealScalar> Load<Block<S, WIDTH>> for Interleaved<&[Complex<S>]> {
    #[inline(always)]
    fn load(&self, i: usize) -> Block<S, WIDTH> {
        let at = &self.0[i * WIDTH..][..WIDTH];
        Block::new(
            Reals::from_fn(|lane| at[lane].re),
            Reals::from_fn(|lane| at[lane].im),
        )
    }
}

impl<S: RealScalar> Store<Block<S, WIDTH>> for Interleaved<&mut [Complex<S>]> {
    #[inline(always)]
    fn store(&mut self, i: usize, value: Block<S, WIDTH>) {
        for (lane, slot) in self.0[i * WIDTH..][..WIDTH].iter_mut().enumerate() {
            *slot = value.lane(lane);
        }
    }
}

/// Copy elements `0..run.len` of the lanes `block..block + WIDTH` of `run`
/// into interleaved `buffer`.
#[inline(always)]
pub(crate) fn interleave<A: Addressing, T: Copy>(
    values: &[T],
    run: Run,
    block: usize,
    buffer: &mut [T],
) {
    let lanes = A::read(values, run, block);
    for (i, slot) in buffer[..run.len * WIDTH]
        .chunks_exact_mut(WIDTH)
        .enumerate()
    {
        slot.copy_from_slice(&lanes.read(i));
    }
}

/// Inverse of [`interleave`].
#[inline(always)]
pub(crate) fn deinterleave<A: Addressing, T: Copy>(
    buffer: &[T],
    values: &mut [T],
    run: Run,
    block: usize,
) {
    let mut lanes = A::write(values, run, block);
    for (i, slot) in buffer[..run.len * WIDTH].chunks_exact(WIDTH).enumerate() {
        lanes.write(i, core::array::from_fn(|lane| slot[lane]));
    }
}

/// Rows of block values read in order.
impl<T: Copy> Load<T> for &[T] {
    #[inline(always)]
    fn load(&self, i: usize) -> T {
        self[i]
    }
}

/// Rows of block values written in order.
impl<T: Copy> Store<T> for &mut [T] {
    #[inline(always)]
    fn store(&mut self, i: usize, value: T) {
        self[i] = value;
    }
}

/// [`interleave`] for the first `count < WIDTH` lanes of a block; the other
/// lanes are `fill`.
pub(crate) fn interleave_partial<T: Copy>(
    values: &[T],
    (run, block, count): (Run, usize, usize),
    buffer: &mut [T],
    fill: T,
) {
    let first = run.first + block * run.step;
    for (i, slot) in buffer[..run.len * WIDTH]
        .chunks_exact_mut(WIDTH)
        .enumerate()
    {
        for (lane, value) in slot.iter_mut().enumerate() {
            *value = if lane < count {
                values[first + lane * run.step + i * run.stride]
            } else {
                fill
            };
        }
    }
}

/// [`deinterleave`] of the first `count < WIDTH` lanes of a block.
pub(crate) fn deinterleave_partial<T: Copy>(
    buffer: &[T],
    values: &mut [T],
    (run, block, count): (Run, usize, usize),
) {
    let first = run.first + block * run.step;
    for (i, slot) in buffer[..run.len * WIDTH].chunks_exact(WIDTH).enumerate() {
        for (lane, &value) in slot[..count].iter().enumerate() {
            values[first + lane * run.step + i * run.stride] = value;
        }
    }
}

/// [`WIDTH`] lanes of a run at explicit starts. Lanes past `count` repeat the
/// last real lane, so a partial block computes only finite copies and every
/// lane takes the same arithmetic wherever it sits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockLanes {
    starts: [usize; WIDTH],
    stride: usize,
    count: usize,
}

impl BlockLanes {
    pub(crate) fn new(run: Run, block: usize) -> Self {
        let count = (run.count - block).min(WIDTH);
        Self {
            starts: core::array::from_fn(|lane| {
                run.first + (block + lane.min(count - 1)) * run.step
            }),
            stride: run.stride,
            count,
        }
    }

    #[inline(always)]
    pub(crate) fn read<T: Copy>(&self, values: &[T], i: usize) -> [T; WIDTH] {
        core::array::from_fn(|lane| values[self.starts[lane] + i * self.stride])
    }

    #[inline(always)]
    pub(crate) fn write<T: Copy>(&self, values: &mut [T], i: usize, lanes: [T; WIDTH]) {
        for (lane, value) in lanes.into_iter().enumerate().take(self.count) {
            values[self.starts[lane] + i * self.stride] = value;
        }
    }
}

/// One real value of each of `L` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reals<S, const L: usize>(pub(crate) [S; L]);

impl<S: RealScalar, const L: usize> Reals<S, L> {
    #[inline(always)]
    pub(crate) fn splat(value: S) -> Self {
        Self([value; L])
    }

    #[inline(always)]
    pub(crate) fn from_fn(value: impl FnMut(usize) -> S) -> Self {
        Self(core::array::from_fn(value))
    }

    /// Restore with one rounding: exact zeros become positive, other values
    /// are multiplied by their lane's binary factor.
    #[inline(always)]
    pub(crate) fn restore(self, factor: Self) -> Self {
        Self::from_fn(|lane| {
            if self.0[lane] == S::ZERO {
                S::ZERO
            } else {
                self.0[lane] * factor.0[lane]
            }
        })
    }

    /// The larger of `self` and `|value|` in each lane; any NaN in `value`
    /// is ignored, so callers check finiteness separately.
    #[inline(always)]
    pub(crate) fn max_abs(self, value: Self) -> Self {
        Self::from_fn(|lane| {
            let magnitude = value.0[lane].abs();
            if magnitude > self.0[lane] {
                magnitude
            } else {
                self.0[lane]
            }
        })
    }

    /// Whether every lane is finite, without short-circuiting.
    #[inline(always)]
    pub(crate) fn finite(self) -> bool {
        self.0
            .iter()
            .fold(true, |finite, value| finite & value.is_finite())
    }
}

impl<S: RealScalar, const L: usize> Add for Reals<S, L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|lane| self.0[lane] + rhs.0[lane])
    }
}

impl<S: RealScalar, const L: usize> Sub for Reals<S, L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_fn(|lane| self.0[lane] - rhs.0[lane])
    }
}

impl<S: RealScalar, const L: usize> Neg for Reals<S, L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_fn(|lane| -self.0[lane])
    }
}

impl<S: RealScalar, const L: usize> Mul for Reals<S, L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_fn(|lane| self.0[lane] * rhs.0[lane])
    }
}

impl<S: RealScalar, const L: usize> Mul<S> for Reals<S, L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: S) -> Self {
        Self::from_fn(|lane| self.0[lane] * rhs)
    }
}

/// One complex value of each of `L` lanes, as component arrays.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Block<S, const L: usize> {
    pub(crate) re: Reals<S, L>,
    pub(crate) im: Reals<S, L>,
}

impl<S: RealScalar, const L: usize> Block<S, L> {
    #[inline(always)]
    pub(crate) const fn new(re: Reals<S, L>, im: Reals<S, L>) -> Self {
        Self { re, im }
    }

    /// Lane `l` is `value(l)`.
    #[inline(always)]
    pub(crate) fn from_fn(mut value: impl FnMut(usize) -> Complex<S>) -> Self {
        let values: [Complex<S>; L] = core::array::from_fn(&mut value);
        Self::new(
            Reals::from_fn(|lane| values[lane].re),
            Reals::from_fn(|lane| values[lane].im),
        )
    }

    /// `Complex::new(re, 0)` in every lane.
    #[inline(always)]
    pub(crate) fn real(re: Reals<S, L>) -> Self {
        Self::new(re, Reals::splat(S::ZERO))
    }

    /// Lane `lane` as one complex value.
    #[inline(always)]
    pub(crate) fn lane(self, lane: usize) -> Complex<S> {
        Complex::new(self.re.0[lane], self.im.0[lane])
    }

    /// [`Complex::conj`] in every lane.
    #[inline(always)]
    pub(crate) fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl<S: RealScalar, const L: usize> super::scalar::Butterfly<S> for Block<S, L> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(Reals::splat(S::ZERO), Reals::splat(S::ZERO))
    }

    #[inline(always)]
    fn bounded(self) -> Self {
        let valid = (0..L).fold(true, |valid, lane| {
            valid & (self.re.0[lane].abs() <= S::ONE) & (self.im.0[lane].abs() <= S::ONE)
        });
        assert!(
            valid,
            "bounded FFT input must have finite components with magnitude at most one"
        );
        self
    }

    #[inline(always)]
    fn rotate<const CONJUGATE: bool>(self, root: Complex<S>) -> Self {
        self * if CONJUGATE { root.conj() } else { root }
    }
    #[inline(always)]
    fn rotate_fused<const CONJUGATE: bool>(self, root: Complex<S>) -> Self {
        Self::from_fn(|lane| self.lane(lane).rotate_fused::<CONJUGATE>(root))
    }

    #[inline(always)]
    fn quarter<const INVERSE: bool>(self) -> Self {
        if INVERSE {
            Self::new(-self.im, self.re)
        } else {
            Self::new(self.im, -self.re)
        }
    }
}

impl<S: RealScalar> From<Complex<S>> for Block<S, 1> {
    #[inline(always)]
    fn from(value: Complex<S>) -> Self {
        Self::new(Reals([value.re]), Reals([value.im]))
    }
}

impl<S: RealScalar, const L: usize> Add for Block<S, L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<S: RealScalar, const L: usize> Sub for Block<S, L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<S: RealScalar, const L: usize> Mul<S> for Block<S, L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: S) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

/// The complex product with a value shared by every lane. IEEE products and
/// sums commute, so this also equals `rhs * self` lane by lane.
impl<S: RealScalar, const L: usize> Mul<Complex<S>> for Block<S, L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Complex<S>) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn bits<S: RealScalar>(value: Complex<S>) -> (u64, u64) {
        (
            crate::test_support::wide(value.re).to_bits(),
            crate::test_support::wide(value.im).to_bits(),
        )
    }

    fn lanes_follow_scalar_complex_arithmetic<S: RealScalar>() {
        let values = [
            S::ZERO,
            -S::ZERO,
            S::MIN_SUBNORMAL,
            -S::MIN_SUBNORMAL,
            S::ONE,
            -S::HALF,
            S::MAX.scalbn(-24),
            -S::MAX.scalbn(-24),
            S::MIN_POSITIVE.scalbn(-5),
        ];
        let complex: Vec<_> = values
            .iter()
            .flat_map(|&re| values.iter().map(move |&im| Complex::new(re, im)))
            .collect();
        let root = Complex::new(S::HALF.next_after(S::ONE), -S::ONE.scalbn(-3));
        for window in complex.windows(WIDTH).step_by(3) {
            for other in complex.windows(WIDTH).step_by(5) {
                let (a, b) = (
                    Block::<S, WIDTH>::from_fn(|lane| window[lane]),
                    Block::<S, WIDTH>::from_fn(|lane| other[lane]),
                );
                let scale = S::ONE.scalbn(-2);
                for (lane, (&x, &y)) in window.iter().zip(other).enumerate() {
                    assert_eq!(bits((a + b).lane(lane)), bits(x + y));
                    assert_eq!(bits((a - b).lane(lane)), bits(x - y));
                    assert_eq!(bits((a * scale).lane(lane)), bits(x * scale));
                    assert_eq!(bits((a * root).lane(lane)), bits(x * root));
                    assert_eq!(bits((a * root).lane(lane)), bits(root * x));
                    assert_eq!(bits(a.conj().lane(lane)), bits(x.conj()));
                    let one = Block::from(x);
                    assert_eq!(bits((one * root).lane(0)), bits(x * root));
                }
                let factors = Reals::<S, WIDTH>::from_fn(|lane| S::ONE.scalbn(lane as i32 - 2));
                let restored = a.re.restore(factors);
                for (lane, value) in window.iter().enumerate() {
                    let expected =
                        crate::scaling::BinaryScale::new(lane as i32 - 2).restore(value.re);
                    assert_eq!(
                        crate::test_support::wide(restored.0[lane]).to_bits(),
                        crate::test_support::wide(expected).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn block_lanes_equal_scalar_complex_operations() {
        lanes_follow_scalar_complex_arithmetic::<f32>();
        lanes_follow_scalar_complex_arithmetic::<f64>();
    }
}
