//! Gohberg–Kailath–Olshevsky elimination (Math. Comp. 64, 1995, §§2–3).
//!
//! For the cyclic shifts Z₊ and Z₋, Z₊T - TZ₋ = GH has rank two.
//! With F the unitary negative DFT and D[j,j] = exp(πij/n),
//! C = FTD F* has entries (g_i h_j)/(t_i-s_j), where t and s are
//! the roots of +1 and -1. Row pivoting preserves this displacement.
//! Eliminate transformed RHS columns alongside the generators: only U is stored.
#![expect(
    clippy::needless_range_loop,
    reason = "indexed in-place recurrences share overlapping numerical storage"
)]

use super::*;
use crate::checked::finite_candidate;
use crate::kernel::gko::{complex, get, put, reconstruct_line, subtract_product, width};
use crate::kernel::standard::{exponent, magnitude, scale};
use scioxide_core::Complex;
use scioxide_core::backend::{checked_add, checked_product};

fn swap<T: NumericScalar>(values: &mut [T], i: usize, j: usize) {
    let a = get(values, i);
    let b = get(values, j);
    put(values, i, b);
    put(values, j, a);
}

impl<T: NumericScalar> StructuredSolvePlan<'_, T> {
    pub(super) fn gko_len(&self) -> Result<usize, SciOxideError> {
        let n = self.order();
        if n <= 1 || self.compact == Compact::Circulant {
            return Ok(0);
        }
        let q = scioxide_core::quantity!("gko_workspace");
        let successor = checked_add(n, 1, q)?;
        let triangle = if n.is_multiple_of(2) {
            checked_product(&[n / 2, successor], q)?
        } else {
            checked_product(&[n, successor / 2], q)?
        };
        checked_product(
            &[
                width::<T>(),
                checked_add(triangle, checked_product(&[7, n], q)?, q)?,
            ],
            q,
        )
    }

    pub(super) fn prepare_gko_phases(
        &mut self,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        let denominator = crate::checked::rounded_count::<T::Real>(n)?;
        for k in 0..=n {
            execution.check()?;
            let angle = crate::checked::rounded_count::<T::Real>(k)? / denominator;
            let (s, c) = scioxide_core::sin_cos_pi(angle)?;
            put(&mut self.phases, k, Complex::new(c, s));
            if k > 0 && k < n {
                put(&mut self.phases, 2 * n - k, Complex::new(c, -s));
            }
        }
        Ok(())
    }

    // Direct DFT with table lookup and wrapping addition, not a trigonometric
    // evaluation or a floating-point phase recurrence for each term.
    fn dft(
        &self,
        step: usize,
        conjugate: bool,
        value: impl Fn(usize) -> Complex<T::Real>,
    ) -> Complex<T::Real> {
        let n = self.order();
        let mut phase = 0;
        let mut sums = [Complex::new(T::Real::ZERO, T::Real::ZERO); 4];
        let mut accumulate = |j: usize, sum: &mut Complex<T::Real>| {
            let root = get(&self.phases, phase);
            let root = if conjugate { root.conj() } else { root };
            let input = value(j);
            // Forward inputs are real for real T; inverse projection has its
            // own paired-frequency loop below. Avoid multiplying known zeros.
            let term = if T::IS_REAL {
                Complex::new(input.re * root.re, input.re * root.im)
            } else {
                input * root
            };
            *sum += term;
            phase += step;
            if phase >= 2 * n {
                phase -= 2 * n;
            }
        };
        for block in 0..n / 4 {
            for (lane, sum) in sums.iter_mut().enumerate() {
                accumulate(4 * block + lane, sum);
            }
        }
        for (lane, sum) in sums[..n % 4].iter_mut().enumerate() {
            accumulate(n - n % 4 + lane, sum);
        }
        (sums[0] + sums[1]) + (sums[2] + sums[3])
    }

    /// Try all pending columns. Accepted columns keep their solution; rejected
    /// ones stay marked for the dense fallback, if it was reserved.
    pub(super) fn try_gko(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize, usize) -> T,
        scratch: &mut [T],
        storage: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let stride = self.stride()?;
        if self.order() <= 1
            || self.compact == Compact::Circulant
            || !scratch
                .chunks_exact(stride)
                .any(|w| !ColumnOutcome::load(&w[stride - 2..]).is_accepted())
        {
            return Ok(());
        }
        match self.gko(op, &rhs, scratch, storage, execution) {
            Err(error)
                if error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability =>
            {
                Ok(())
            }
            result => result,
        }
    }

    fn gko(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize, usize) -> T,
        scratch: &mut [T],
        storage: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        let dispatch = crate::KernelContext::new(execution).dispatch();
        let stride = self.stride()?;
        let wn = width::<T>() * n;
        let (g0, rest) = storage.split_at_mut(wn);
        let (g1, rest) = rest.split_at_mut(wn);
        let (h0, rest) = rest.split_at_mut(wn);
        let (h1, rest) = rest.split_at_mut(wn);
        let (nodes, rest) = rest.split_at_mut(wn);
        let (column_nodes, rest) = rest.split_at_mut(wn);
        let (column, upper) = rest.split_at_mut(wn);
        let unit = T::Real::ONE / crate::checked::rounded_count::<T::Real>(n)?.sqrt();
        // Temporary native coefficient vectors; nodes/column are not live yet.
        for j in 0..n {
            nodes[j] = if j == 0 {
                self.entry(op, 0, 0)
            } else {
                self.entry(op, j, 0) + self.entry(op, 0, n - j)
            };
            column[j] = if j + 1 == n {
                self.entry(op, 0, 0)
            } else {
                self.entry(op, n - 1 - j, 0) - self.entry(op, 0, j + 1)
            };
        }
        for k in 0..n {
            execution.check()?;
            put(g0, k, Complex::new(unit, T::Real::ZERO));
            if !T::IS_REAL || k <= n / 2 {
                let value = finite_candidate(self.dft(2 * k, true, |j| complex(nodes[j])) * unit)?;
                put(g1, k, value);
                if T::IS_REAL && k > 0 && 2 * k < n {
                    put(g1, n - k, value.conj());
                }
            }
            if !T::IS_REAL || k < n.div_ceil(2) {
                let value =
                    finite_candidate(self.dft(2 * k + 1, false, |j| complex(column[j])) * unit)?;
                put(h0, k, value);
                if T::IS_REAL && k != n - 1 - k {
                    put(h0, n - 1 - k, value.conj());
                }
            }
            let odd = 2 * k + 1;
            let phase = if odd <= n { n - odd } else { 3 * n - odd };
            put(h1, k, get(&self.phases, phase) * unit);
        }
        for k in 0..n {
            put(nodes, k, get(&self.phases, 2 * k).conj());
            put(column_nodes, k, get(&self.phases, 2 * k + 1).conj());
        }
        for (j, work) in scratch.chunks_exact_mut(stride).enumerate() {
            if ColumnOutcome::load(&work[stride - 2..]).is_accepted() {
                continue;
            }
            let frame = exponent((0..n).fold(T::Real::ZERO, |m, i| m.max(magnitude(rhs(i, j)))))?;
            let (normalized, rest) = work.split_at_mut(n);
            let rest = &mut rest[..wn];
            for (i, value) in normalized.iter_mut().enumerate() {
                *value = scale(rhs(i, j), -frame);
            }
            for k in 0..if T::IS_REAL { n / 2 + 1 } else { n } {
                execution.check()?;
                let value =
                    finite_candidate(self.dft(2 * k, true, |i| complex(normalized[i])) * unit)?;
                put(rest, k, value);
                if T::IS_REAL && k > 0 && 2 * k < n {
                    put(rest, n - k, value.conj());
                }
            }
        }
        let mut offset = 0;
        for k in 0..n {
            execution.check()?;
            let s = get(&self.phases, 2 * k + 1).conj();
            let a = get(h0, k);
            let b = get(h1, k);
            let mut finite =
                reconstruct_line((g0, g1, nodes), (a, b), s, k..n, column, k, dispatch);
            if !finite {
                return Err(super::kernels::instability(
                    "GKO reconstructed a non-finite column",
                ));
            }
            let pivot = crate::kernel::gko::pivot(column, k);
            if get(column, pivot) == Complex::new(T::Real::ZERO, T::Real::ZERO) {
                return Err(SciOxideError::singular(None));
            }
            swap(g0, k, pivot);
            swap(g1, k, pivot);
            swap(nodes, k, pivot);
            swap(column, k, pivot);
            let diagonal = get(column, k);
            let inverse = finite_candidate(Complex::new(T::Real::ONE, T::Real::ZERO) / diagonal)?;
            let a = get(g0, k);
            let b = get(g1, k);
            let t = get(nodes, k);
            let hk0 = get(h0, k);
            let hk1 = get(h1, k);
            put(upper, offset, diagonal);
            finite &= reconstruct_line(
                (h0, h1, column_nodes),
                (-a, -b),
                t,
                k + 1..n,
                upper,
                offset + 1,
                dispatch,
            );
            let remaining = n - k - 1;
            finite &= subtract_product(
                h0,
                k + 1,
                upper,
                offset + 1,
                remaining,
                inverse * hk0,
                dispatch,
            );
            finite &= subtract_product(
                h1,
                k + 1,
                upper,
                offset + 1,
                remaining,
                inverse * hk1,
                dispatch,
            );
            finite &= subtract_product(g0, k + 1, column, k + 1, remaining, inverse * a, dispatch);
            finite &= subtract_product(g1, k + 1, column, k + 1, remaining, inverse * b, dispatch);
            for work in scratch.chunks_exact_mut(stride) {
                if ColumnOutcome::load(&work[stride - 2..]).is_accepted() {
                    continue;
                }
                let y = &mut work[n..n + wn];
                swap(y, k, pivot);
                let b = get(y, k);
                finite &=
                    subtract_product(y, k + 1, column, k + 1, remaining, inverse * b, dispatch);
            }
            if !finite {
                return Err(super::kernels::instability(
                    "GKO generator or RHS update is non-finite",
                ));
            }
            offset += n - k;
        }
        for (j, work) in scratch.chunks_exact_mut(stride).enumerate() {
            if ColumnOutcome::load(&work[stride - 2..]).is_accepted() {
                continue;
            }
            let (output, rest) = work.split_at_mut(n);
            let y = &mut rest[..wn];
            let mut offset = upper.len() / width::<T>();
            for k in (0..n).rev() {
                execution.check()?;
                offset -= n - k;
                let mut sum = get(y, k);
                let product =
                    crate::kernel::gko::upper_product(upper, offset + 1, y, k + 1, dispatch)?;
                if let Some(product) = product {
                    sum -= product;
                } else {
                    let mut sums = [Complex::new(T::Real::ZERO, T::Real::ZERO); 4];
                    for i in k + 1..n {
                        let lane = (i - k - 1) % 4;
                        sums[lane] += get(upper, offset + i - k) * get(y, i);
                    }
                    sum -= (sums[0] + sums[1]) + (sums[2] + sums[3]);
                }
                put(y, k, finite_candidate(sum / get(upper, offset))?);
            }
            if T::IS_REAL {
                // Pair opposite half-shifted frequencies before projection.
                // This computes the real part even if rounding broke symmetry.
                for k in 0..n / 2 {
                    put(column, k, get(y, k) + get(y, n - 1 - k).conj());
                }
                for i in 0..n {
                    execution.check()?;
                    let mut phase = i;
                    let mut sums = [T::Real::ZERO; 4];
                    let mut accumulate = |k: usize, sum: &mut T::Real| {
                        let value = get(column, k);
                        let root = get(&self.phases, phase);
                        *sum = *sum + value.re * root.re - value.im * root.im;
                        phase += 2 * i;
                        if phase >= 2 * n {
                            phase -= 2 * n;
                        }
                    };
                    for block in 0..(n / 2) / 4 {
                        for (lane, sum) in sums.iter_mut().enumerate() {
                            accumulate(4 * block + lane, sum);
                        }
                    }
                    for (lane, sum) in sums[..(n / 2) % 4].iter_mut().enumerate() {
                        accumulate(n / 2 - (n / 2) % 4 + lane, sum);
                    }
                    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                    if n % 2 == 1 {
                        let middle = get(y, n / 2).re;
                        sum += if i % 2 == 0 { middle } else { -middle };
                    }
                    output[i] = finite_candidate(T::from_real(sum * unit))?;
                }
            } else {
                for i in 0..n {
                    execution.check()?;
                    let z = finite_candidate(
                        self.dft(2 * i, false, |k| get(y, k)) * get(&self.phases, i) * unit,
                    )?;
                    output[i] = T::from_components(z.re, z.im);
                }
            }
            if self.compact == Compact::Hankel {
                output.reverse();
            }
            let frame = exponent((0..n).fold(T::Real::ZERO, |m, i| m.max(magnitude(rhs(i, j)))))?;
            match self
                .restore(output, frame)
                .and_then(|()| self.assess(op, |i| rhs(i, j), output, &mut rest[..n], execution))
            {
                Ok(mut report) => {
                    report.gko = usize::from(report.termination.is_success());
                    ColumnOutcome::Assessed(report).store(&mut work[stride - 2..]);
                }
                Err(error)
                    if error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}
