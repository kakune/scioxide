//! Real-data decimation in time for odd lengths `n = p * m` with `p` in
//! {3, 5, 7}.
//!
//! The samples split into `p` decimated sequences `x_r[t] = x[p t + r]`.
//! Sequences `x_{2q+1} + i x_{2q+2}` share one length-`m` complex FFT per pair
//! and `x_0` takes a recursive odd real plan, so a transform costs about half
//! a complex FFT of length `n`. One twiddled `p`-point butterfly per column
//! `j <= (m-1)/2` recombines the decimated spectra: its outputs `0..=p/2` are
//! bins `j + q m` and the conjugates of the rest are bins `(m - j) + q m`, so
//! every packed bin is written exactly once. The inverse mirrors these steps
//! and is real by construction.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::size_of;

use scioxide_core::Complex;
use scioxide_core::backend::{KernelBackend, checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, ExecutionPolicy, SciOxideError};

use super::RealFftPlan;
use crate::complex::ComplexCore;
use crate::numeric::reduced_root;
use crate::numeric::scalar_from_usize_rounded;
use crate::scaling::BinaryScale;
use crate::types::FftPlanDiagnostics;

/// Disjoint pair, residue and complex-child scratch regions.
type WorkRegions<'a, S> = (
    &'a mut [Complex<S>],
    &'a mut [Complex<S>],
    &'a mut [Complex<S>],
);

/// Decimation radices, in order of preference (cheapest butterfly first).
const RADICES: [usize; 3] = [3, 5, 7];

/// The radix that decimates odd `n`, if `n` is a proper multiple of one.
pub(super) fn radix(n: usize) -> Option<usize> {
    RADICES
        .into_iter()
        .find(|&radix| n > radix && n.is_multiple_of(radix))
}

#[derive(Debug, Clone)]
pub(super) struct Decimation<S: scioxide_core::RealScalar> {
    radix: usize,
    /// Decimated length `m = n / radix`.
    len: usize,
    /// Shared length-`m` FFT of the paired sequences.
    complex: ComplexCore<S>,
    /// Recursive odd real plan of `x_0`.
    residue: Box<RealFftPlan<S>>,
    /// `w_n^(r j)` at `[j * (radix - 1) + r - 1]` for columns `j <= (m-1)/2`.
    twiddles: Vec<Complex<S>>,
    /// `w_radix^s` for `s < radix`.
    roots: [Complex<S>; 7],
    /// Inverse weights keep every child input bounded while matching the
    /// residue's raw gain: pairs receive `leaf / n`, the residue `1 / radix`.
    pair_weight: S,
    residue_weight: S,
    /// `n / leaf`, the factor between this plan's raw inverse and `n * x`.
    gain: S,
}

impl<S: scioxide_core::RealScalar> Decimation<S> {
    pub(super) fn new(
        n: usize,
        radix: usize,
        execution_policy: ExecutionPolicy,
        backend: KernelBackend,
        context: crate::FftPlanningContext<'_, S>,
        length: Argument,
        provider: Argument,
    ) -> Result<Self, SciOxideError> {
        let len = n / radix;
        let complex = ComplexCore::with_backend_and_context(
            len,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?;
        let residue = Box::new(RealFftPlan::with_backend_and_context(
            len,
            execution_policy,
            backend,
            context,
            length,
            provider,
        )?);
        let columns = len.div_ceil(2);
        let mut twiddles = try_with_capacity(
            checked_mul(
                columns,
                radix - 1,
                scioxide_core::quantity!("real_fft_twiddle_count"),
            )?,
            scioxide_core::quantity!("real_fft_twiddles"),
        )?;
        for column in 0..columns {
            for branch in 1..radix {
                twiddles.push(reduced_root::<S>(
                    (branch * column) as u128,
                    n as u128,
                    false,
                    length,
                )?);
            }
        }
        let mut roots = [Complex::new(S::ONE, S::ZERO); 7];
        for (exponent, root) in roots.iter_mut().enumerate().take(radix) {
            *root = reduced_root::<S>(exponent as u128, radix as u128, false, length)?;
        }
        let leaf = residue.leaf_len();
        let n_scalar = scalar_from_usize_rounded::<S>(n)?;
        Ok(Self {
            radix,
            len,
            pair_weight: scalar_from_usize_rounded::<S>(leaf)? / n_scalar,
            residue_weight: S::ONE / scalar_from_usize_rounded::<S>(radix)?,
            gain: scalar_from_usize_rounded::<S>(n / leaf)?,
            complex,
            residue,
            twiddles,
            roots,
        })
    }

    /// Paired sequences occupy `(radix - 1) / 2` blocks of `m` values.
    fn pairs_len(&self) -> usize {
        self.radix / 2 * self.len
    }

    /// Scratch after the packed staging slots: pair blocks, residue scratch
    /// and the complex child's own scratch.
    pub(super) fn work_len(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("real_fft_scratch_length");
        checked_add(
            checked_add(self.pairs_len(), self.residue.scratch_len(), q)?,
            self.complex.bounded_scratch_len(),
            q,
        )
    }

    /// Owned heap bytes: the complex child's tables, the boxed residue and
    /// the twiddles.
    pub(super) fn owned_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("real_fft_owned_plan_bytes");
        checked_add(
            checked_add(
                self.complex.plan_bytes() - size_of::<ComplexCore<S>>(),
                self.residue.plan_bytes(),
                q,
            )?,
            checked_mul(self.twiddles.capacity(), size_of::<Complex<S>>(), q)?,
            q,
        )
    }

    /// The longer of the pair and residue paths, plus the recombination.
    pub(super) fn diagnostics(&self) -> FftPlanDiagnostics {
        let pairs = self.complex.diagnostics();
        let residue = self.residue.diagnostics();
        FftPlanDiagnostics {
            stage_count: pairs.stage_count.max(residue.stage_count),
            ..pairs
        }
        .with_additional_stage()
    }

    pub(super) fn children(&self) -> (&ComplexCore<S>, &RealFftPlan<S>) {
        (&self.complex, &self.residue)
    }

    pub(super) fn backend(&self) -> KernelBackend {
        self.complex.backend()
    }

    pub(super) fn leaf_len(&self) -> usize {
        self.residue.leaf_len()
    }

    pub(super) const fn gain(&self) -> S {
        self.gain
    }

    /// Byte limit of `work_len` storage imposed by provider children.
    pub(super) fn work_byte_limit(&self, work_len: usize) -> usize {
        let bytes = |len: usize| len * size_of::<Complex<S>>();
        crate::workspace::nested_byte_limit(
            usize::MAX,
            bytes(work_len),
            [
                (
                    self.complex.bounded_workspace_byte_limit(),
                    bytes(self.complex.bounded_scratch_len()),
                ),
                (
                    self.residue.workspace_byte_limit(),
                    bytes(self.residue.scratch_len()),
                ),
            ],
        )
    }

    /// Pair blocks, residue scratch and complex-child scratch within `work`.
    fn split<'a>(&self, work: &'a mut [Complex<S>]) -> WorkRegions<'a, S> {
        let (pairs, rest) = work.split_at_mut(self.pairs_len());
        let (residue, complex) = rest.split_at_mut(self.residue.scratch_len());
        (pairs, residue, complex)
    }
    /// Replace the packed samples staged in `data` (finite, <=1) by their
    /// unscaled packed spectrum.
    pub(super) fn forward(
        &self,
        data: &mut [Complex<S>],
        work: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self.radix {
            3 => self.forward_radix::<3>(data, work, context),
            5 => self.forward_radix::<5>(data, work, context),
            _ => self.forward_radix::<7>(data, work, context),
        }
    }

    fn forward_radix<const P: usize>(
        &self,
        data: &mut [Complex<S>],
        work: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        debug_assert_eq!(self.radix, P);
        crate::validation::length(data.len(), (P * self.len) / 2 + 1)?;
        crate::validation::sufficient(work.len(), self.work_len()?)?;
        context.check()?;
        let (pairs, residue, complex_work) = self.split(work);
        self.gather::<P>(data, pairs, residue);
        crate::validation::sufficient(complex_work.len(), self.complex.bounded_scratch_len())?;
        for block in pairs.chunks_exact_mut(self.len) {
            self.complex
                .execute_bounded(block, complex_work, false, S::ONE, context)?;
        }
        let residue = self.residue.forward_bounded(residue, |_| {}, context)?;
        self.combine::<P>(pairs, residue, data);
        context.check()
    }

    /// Split the packed samples `data` into pair blocks and the residue's
    /// staging. Each run of `P` slots holds samples of columns `2u` and
    /// `2u + 1`, so all component offsets are compile-time constants.
    fn gather<const P: usize>(
        &self,
        data: &[Complex<S>],
        pairs: &mut [Complex<S>],
        residue: &mut [Complex<S>],
    ) {
        let m = self.len;
        let packed = self.residue.is_packed();
        let mut bounded = true;
        let mut put = |t: usize, chunk: &[Complex<S>], base: usize| {
            for q in 0..P / 2 {
                let value = Complex::new(
                    component(chunk, base + 2 * q + 1),
                    component(chunk, base + 2 * q + 2),
                );
                bounded &= (value.re.abs() <= S::ONE) & (value.im.abs() <= S::ONE);
                pairs[q * m + t] = value;
            }
        };
        for (u, chunk) in data.as_chunks::<P>().0[..m / 2].iter().enumerate() {
            put(2 * u, chunk, 0);
            put(2 * u + 1, chunk, P);
            let (first, second) = (component(chunk, 0), component(chunk, P));
            if packed {
                residue[u] = Complex::new(first, second);
            } else {
                residue[2 * u] = Complex::new(first, S::ZERO);
                residue[2 * u + 1] = Complex::new(second, S::ZERO);
            }
        }
        // Odd m leaves column m - 1, whose first sample is even.
        let t = m - 1;
        let chunk = &data[P * t / 2..];
        put(t, chunk, 0);
        self.residue.stage_value(residue, t, component(chunk, 0));
        assert!(
            bounded,
            "bounded FFT input must have finite components with magnitude at most one"
        );
    }

    /// Inverse of [`Self::gather`]: interleave the pair blocks and the
    /// residue's staged samples into packed `data`.
    fn scatter<const P: usize>(
        &self,
        data: &mut [Complex<S>],
        pairs: &[Complex<S>],
        residue: &[Complex<S>],
    ) {
        let m = self.len;
        let packed = self.residue.is_packed();
        let put = |t: usize, chunk: &mut [Complex<S>], base: usize| {
            for q in 0..P / 2 {
                let value = pairs[q * m + t];
                *component_mut(chunk, base + 2 * q + 1) = value.re;
                *component_mut(chunk, base + 2 * q + 2) = value.im;
            }
        };
        for (u, chunk) in data.as_chunks_mut::<P>().0[..m / 2].iter_mut().enumerate() {
            put(2 * u, chunk, 0);
            put(2 * u + 1, chunk, P);
            let (first, second) = if packed {
                (residue[u].re, residue[u].im)
            } else {
                (residue[2 * u].re, residue[2 * u + 1].re)
            };
            *component_mut(chunk, 0) = first;
            *component_mut(chunk, P) = second;
        }
        let t = m - 1;
        let chunk = &mut data[P * t / 2..];
        put(t, chunk, 0);
        *component_mut(chunk, 0) = self.residue.staged_value(residue, t);
        // Odd n leaves the last packed slot's imaginary part unused.
        chunk[P / 2].im = S::ZERO;
    }

    fn combine<const P: usize>(
        &self,
        pairs: &[Complex<S>],
        residue: &[Complex<S>],
        output: &mut [Complex<S>],
    ) {
        debug_assert_eq!(self.radix, P);
        let m = self.len;
        let half = P / 2;
        for (column, twiddles) in self.twiddles.chunks_exact(P - 1).enumerate() {
            let mirror = (m - column) % m;
            let mut y = [Complex::new(S::ZERO, S::ZERO); P];
            y[0] = residue[column];
            for (q, block) in pairs.chunks_exact(m).enumerate() {
                let (a, b) = (block[column], block[mirror].conj());
                // X_{2q+1} = (a + b) / 2 and X_{2q+2} = -i (a - b) / 2.
                let first = (a + b) * S::HALF;
                let second = Complex::new(a.im - b.im, b.re - a.re) * S::HALF;
                y[2 * q + 1] = first * twiddles[2 * q];
                y[2 * q + 2] = second * twiddles[2 * q + 1];
            }
            let out = butterfly(&y, &self.roots);
            for q in 0..=half {
                output[column + q * m] = out[q];
            }
            if column > 0 {
                for q in 0..half {
                    output[m - column + q * m] = out[P - 1 - q].conj();
                }
            }
        }
    }

    /// Replace the packed spectrum in `data` (components scaled by `scaling`
    /// are finite and at most 1/4) by `n / gain` times its real samples, in
    /// packed staging order.
    pub(super) fn inverse(
        &self,
        data: &mut [Complex<S>],
        work: &mut [Complex<S>],
        scaling: BinaryScale<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self.radix {
            3 => self.inverse_radix::<3>(data, work, scaling, context),
            5 => self.inverse_radix::<5>(data, work, scaling, context),
            _ => self.inverse_radix::<7>(data, work, scaling, context),
        }
    }

    fn inverse_radix<const P: usize>(
        &self,
        data: &mut [Complex<S>],
        work: &mut [Complex<S>],
        scaling: BinaryScale<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        debug_assert_eq!(self.radix, P);
        crate::validation::length(data.len(), (P * self.len) / 2 + 1)?;
        crate::validation::sufficient(work.len(), self.work_len()?)?;
        context.check()?;
        let (pairs, residue, complex_work) = self.split(work);
        crate::validation::sufficient(complex_work.len(), self.complex.bounded_scratch_len())?;
        self.decompose::<P>(data, pairs, residue, scaling);
        for block in pairs.chunks_exact_mut(self.len) {
            self.complex
                .execute_bounded(block, complex_work, true, S::ONE, context)?;
        }
        self.residue
            .inverse_odd_raw(residue, BinaryScale::Identity, context)?;
        self.scatter::<P>(data, pairs, residue);
        context.check()
    }

    fn decompose<const P: usize>(
        &self,
        spectrum: &[Complex<S>],
        pairs: &mut [Complex<S>],
        residue: &mut [Complex<S>],
        scaling: BinaryScale<S>,
    ) {
        debug_assert_eq!(self.radix, P);
        let mut bounded = true;
        let mut put = |destination: &mut Complex<S>, value: Complex<S>| {
            bounded &= (value.re.abs() <= S::ONE) & (value.im.abs() <= S::ONE);
            *destination = value;
        };
        let m = self.len;
        let half = P / 2;
        let scaled =
            |value: Complex<S>| Complex::new(scaling.apply(value.re), scaling.apply(value.im));
        for (column, twiddles) in self.twiddles.chunks_exact(P - 1).enumerate() {
            // Bins j + q m beyond the packed half are conjugates of
            // (P - q) m - j.
            let mut x = [Complex::new(S::ZERO, S::ZERO); P];
            for q in 0..=half {
                x[q] = scaled(spectrum[column + q * m]);
            }
            for q in half + 1..P {
                x[q] = scaled(spectrum[(P - q) * m - column]).conj();
            }
            // The inverse butterfly is the forward one with outputs reversed.
            let v = butterfly(&x, &self.roots);
            let inverse = |r: usize| v[(P - r) % P];
            let mut first = inverse(0) * self.residue_weight;
            if column == 0 {
                first.im = S::ZERO;
            }
            residue[column] = first;
            let weight = self.pair_weight;
            for (q, block) in pairs.chunks_exact_mut(m).enumerate() {
                let even = inverse(2 * q + 1) * twiddles[2 * q].conj() * weight;
                let odd = inverse(2 * q + 2) * twiddles[2 * q + 1].conj() * weight;
                if column == 0 {
                    put(&mut block[0], Complex::new(even.re, odd.re));
                } else {
                    // Z[j] = X_e[j] + i X_o[j]; Z[m-j] uses conj(X[j]).
                    put(
                        &mut block[column],
                        Complex::new(even.re - odd.im, even.im + odd.re),
                    );
                    put(
                        &mut block[m - column],
                        Complex::new(even.re + odd.im, odd.re - even.im),
                    );
                }
            }
        }
        assert!(
            bounded,
            "bounded FFT input must have finite components with magnitude at most one"
        );
    }
}

/// Forward `P`-point DFT of `y` with roots `w_P^s`. Each conjugate output
/// pair shares the symmetric sums `y_r + y_{P-r}` and differences
/// `y_r - y_{P-r}`.
#[inline(always)]
fn butterfly<S: scioxide_core::RealScalar, const P: usize>(
    y: &[Complex<S>; P],
    roots: &[Complex<S>; 7],
) -> [Complex<S>; P] {
    let half = P / 2;
    let mut sums = [Complex::new(S::ZERO, S::ZERO); P];
    let mut differences = [Complex::new(S::ZERO, S::ZERO); P];
    let mut out = [y[0]; P];
    for r in 1..=half {
        sums[r] = y[r] + y[P - r];
        differences[r] = y[r] - y[P - r];
        out[0] += sums[r];
    }
    for k in 1..=half {
        let mut even = y[0];
        let mut odd = Complex::new(S::ZERO, S::ZERO);
        for r in 1..=half {
            let root = roots[r * k % P];
            even += sums[r] * root.re;
            odd += differences[r] * root.im;
        }
        // w^(rk) y_r + w^(-rk) y_{P-r} = re(w) sum + i im(w) difference.
        let rotated = Complex::new(-odd.im, odd.re);
        out[k] = even + rotated;
        out[P - k] = even - rotated;
    }
    out
}

/// Packed sample `offset` of `chunk`: slot `offset / 2`, real part first.
#[inline(always)]
fn component<S: scioxide_core::RealScalar>(chunk: &[Complex<S>], offset: usize) -> S {
    let slot = chunk[offset / 2];
    if offset.is_multiple_of(2) {
        slot.re
    } else {
        slot.im
    }
}

#[inline(always)]
fn component_mut<S: scioxide_core::RealScalar>(chunk: &mut [Complex<S>], offset: usize) -> &mut S {
    let slot = &mut chunk[offset / 2];
    if offset.is_multiple_of(2) {
        &mut slot.re
    } else {
        &mut slot.im
    }
}
