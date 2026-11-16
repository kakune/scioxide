//! Portable scalar FFT codelets.

use scioxide_core::Complex;

use super::{FinalStageKernel, GatherKernel, PermutationKernel, StageKernel, StageKernels};

pub(crate) const fn kernels<S: scioxide_core::RealScalar>() -> StageKernels<S> {
    StageKernels {
        forward: StageKernel::new(forward::<S>),
        inverse: StageKernel::new(inverse::<S>),
        fused_gather_forward: None,
        fused_gather_inverse: None,
        fused_stage_forward: None,
        fused_stage_inverse: None,
        final_forward: FinalStageKernel::new(final_forward::<S>),
        final_inverse: FinalStageKernel::new(final_inverse::<S>),
        permute: PermutationKernel::new(permute::<S, false>, permute::<S, true>),
        gather_forward: GatherKernel::new(gather::<S, false, false>, gather::<S, false, true>),
        gather_inverse: GatherKernel::new(gather::<S, true, false>, gather::<S, true, true>),
    }
}

pub(crate) fn gather<S: scioxide_core::RealScalar, const INVERSE: bool, const BOUNDED: bool>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    permutation: &[usize],
    radix: usize,
    roots: &[Complex<S>],
) {
    match radix {
        2 => gather_radix::<S, 2, INVERSE, BOUNDED>(input, output, permutation, roots),
        3 => gather_radix::<S, 3, INVERSE, BOUNDED>(input, output, permutation, roots),
        4 => gather_radix::<S, 4, INVERSE, BOUNDED>(input, output, permutation, roots),
        5 => gather_radix::<S, 5, INVERSE, BOUNDED>(input, output, permutation, roots),
        7 => gather_radix::<S, 7, INVERSE, BOUNDED>(input, output, permutation, roots),
        _ => unreachable!(),
    }
}

fn gather_radix<
    S: scioxide_core::RealScalar,
    const RADIX: usize,
    const INVERSE: bool,
    const BOUNDED: bool,
>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    permutation: &[usize],
    roots: &[Complex<S>],
) {
    // First stage (width 1) fused with the digit-reversal gather: every
    // twiddle of a width-1 stage is 1, so each group of RADIX gathered inputs
    // only needs the bare codelet. `values` is sized for the largest radix
    // (7) so all codelets share one stack buffer type.
    output
        .chunks_exact_mut(RADIX)
        .zip(permutation.chunks_exact(RADIX))
        .for_each(|(destination, sources)| {
            let mut values = [Complex::new(S::ZERO, S::ZERO); 7];
            for branch in 0..RADIX {
                let value = input[sources[branch]];
                values[branch] = if BOUNDED {
                    crate::validation::bounded(value)
                } else {
                    value
                };
            }
            codelet::<S, _, RADIX, INVERSE>(&mut values, roots);
            destination.copy_from_slice(&values[..RADIX]);
        });
}

pub(crate) fn final_forward<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
) {
    execute_final::<S, false>(input, output, radix, width, twiddles, roots, scale);
}

pub(crate) fn final_inverse<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
) {
    execute_final::<S, true>(input, output, radix, width, twiddles, roots, scale);
}

fn execute_final<S: scioxide_core::RealScalar, const INVERSE: bool>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
) {
    match radix {
        2 => final_radix::<S, 2, INVERSE>(input, output, width, twiddles, roots, scale),
        3 => final_radix::<S, 3, INVERSE>(input, output, width, twiddles, roots, scale),
        4 => final_radix::<S, 4, INVERSE>(input, output, width, twiddles, roots, scale),
        5 => final_radix::<S, 5, INVERSE>(input, output, width, twiddles, roots, scale),
        7 => final_radix::<S, 7, INVERSE>(input, output, width, twiddles, roots, scale),
        _ => unreachable!(),
    }
}

fn final_radix<S: scioxide_core::RealScalar, const RADIX: usize, const INVERSE: bool>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
) {
    // Same butterfly as `butterfly`, but reading from the ping-pong buffer and
    // writing the direction-scaled result to the destination, which saves
    // both a copy-back and a separate scaling pass.
    let block_size = RADIX * width;
    input
        .chunks_exact(block_size)
        .zip(output.chunks_exact_mut(block_size))
        .for_each(|(input_block, output_block)| {
            for index in 0..width {
                let mut values = [Complex::new(S::ZERO, S::ZERO); 7];
                values[0] = input_block[index];
                for branch in 1..RADIX {
                    let twiddle = twiddles[(branch - 1) * width + index];
                    values[branch] = if INVERSE {
                        input_block[branch * width + index] * twiddle.conj()
                    } else {
                        input_block[branch * width + index] * twiddle
                    };
                }
                codelet::<S, _, RADIX, INVERSE>(&mut values, roots);
                for branch in 0..RADIX {
                    let value = values[branch] * scale;
                    output_block[branch * width + index] = value;
                }
            }
        });
}

pub(crate) fn permute<S: scioxide_core::RealScalar, const BOUNDED: bool>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    permutation: &[usize],
) {
    output
        .iter_mut()
        .zip(permutation.iter().copied())
        .for_each(|(destination, source)| {
            let value = input[source];
            *destination = if BOUNDED {
                crate::validation::bounded(value)
            } else {
                value
            };
        });
}

fn forward<S: scioxide_core::RealScalar>(
    data: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) {
    execute::<S, false>(data, radix, width, twiddles, roots);
}

fn inverse<S: scioxide_core::RealScalar>(
    data: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) {
    execute::<S, true>(data, radix, width, twiddles, roots);
}

pub(super) fn execute<S: scioxide_core::RealScalar, const INVERSE: bool>(
    data: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) {
    match radix {
        2 => execute_radix::<S, 2, INVERSE>(data, width, twiddles, roots),
        3 => execute_radix::<S, 3, INVERSE>(data, width, twiddles, roots),
        4 => execute_radix::<S, 4, INVERSE>(data, width, twiddles, roots),
        5 => execute_radix::<S, 5, INVERSE>(data, width, twiddles, roots),
        7 => execute_radix::<S, 7, INVERSE>(data, width, twiddles, roots),
        _ => unreachable!(),
    }
}

fn execute_radix<S: scioxide_core::RealScalar, const RADIX: usize, const INVERSE: bool>(
    data: &mut [Complex<S>],
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) {
    let block_size = RADIX * width;
    data.chunks_exact_mut(block_size).for_each(|block| {
        for index in 0..width {
            butterfly::<S, RADIX, INVERSE>(block, width, index, twiddles, roots);
        }
    });
}

#[inline(always)]
pub(crate) fn butterfly<S: scioxide_core::RealScalar, const RADIX: usize, const INVERSE: bool>(
    block: &mut [Complex<S>],
    width: usize,
    index: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) {
    let values = butterfly_values::<S, RADIX, INVERSE>(block, width, index, twiddles, roots);
    for branch in 0..RADIX {
        block[index + branch * width] = values[branch];
    }
}

#[inline(always)]
fn butterfly_values<S: scioxide_core::RealScalar, const RADIX: usize, const INVERSE: bool>(
    block: &[Complex<S>],
    width: usize,
    index: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
) -> [Complex<S>; 7] {
    // Decimation-in-time butterfly: inputs `index + branch * width` are the
    // branch-th sub-transforms' bin `index`, multiplied by w^(branch * index)
    // before the RADIX-point DFT. Twiddles are stored for the forward sign,
    // and the inverse uses their conjugates. Index 0 has unit twiddles, so the
    // multiplication is skipped.
    let mut values = [Complex::new(S::ZERO, S::ZERO); 7];
    values[0] = block[index];
    for branch in 1..RADIX {
        let value = block[index + branch * width];
        values[branch] = if index == 0 {
            value
        } else {
            let twiddle = twiddles[(branch - 1) * width + index];
            value * if INVERSE { twiddle.conj() } else { twiddle }
        };
    }
    codelet::<S, _, RADIX, INVERSE>(&mut values, roots);
    values
}

/// Arithmetic the radix codelets need from one value: a complex number for
/// single-lane leaves or a [`super::block::Block`] of lanes for across-lane
/// execution. Every implementation performs the same operations in the same
/// order, so one codelet definition serves both.
pub(crate) trait Butterfly<S: scioxide_core::RealScalar>:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<S, Output = Self>
{
    fn zero() -> Self;
    /// Accept finite components of magnitude at most one in an existing read.
    fn bounded(self) -> Self;
    /// Multiply by `root`, or by its conjugate when `CONJUGATE` is set.
    fn rotate<const CONJUGATE: bool>(self, root: Complex<S>) -> Self;
    /// The AVX2 complex product: round the cross product, then fuse the
    /// other product with its signed addend. Used only with verified FMA.
    fn rotate_fused<const CONJUGATE: bool>(self, root: Complex<S>) -> Self;
    /// Multiply by -i (forward) or +i (inverse) as an exact swap and negation.
    fn quarter<const INVERSE: bool>(self) -> Self;
}

impl<S: scioxide_core::RealScalar> Butterfly<S> for Complex<S> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(S::ZERO, S::ZERO)
    }
    #[inline(always)]
    fn bounded(self) -> Self {
        crate::validation::bounded(self)
    }
    #[inline(always)]
    fn rotate<const CONJUGATE: bool>(self, root: Complex<S>) -> Self {
        self * if CONJUGATE { root.conj() } else { root }
    }
    #[inline(always)]
    fn rotate_fused<const CONJUGATE: bool>(self, root: Complex<S>) -> Self {
        let im = if CONJUGATE { -root.im } else { root.im };
        Self::new(
            root.re.mul_add(self.re, -(im * self.im)),
            root.re.mul_add(self.im, im * self.re),
        )
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

#[inline(always)]
pub(crate) fn codelet<
    S: scioxide_core::RealScalar,
    V: Butterfly<S>,
    const RADIX: usize,
    const INVERSE: bool,
>(
    values: &mut [V; 7],
    roots: &[Complex<S>],
) {
    match RADIX {
        2 => radix2(values),
        3 => radix3::<S, V, INVERSE>(values, roots),
        4 => radix4::<S, V, INVERSE>(values),
        5 => radix5::<S, V, INVERSE>(values, roots),
        7 => radix7::<S, V, INVERSE>(values, roots),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn radix2<S: scioxide_core::RealScalar, V: Butterfly<S>>(values: &mut [V; 7]) {
    let a = values[0];
    let b = values[1];
    values[0] = a + b;
    values[1] = a - b;
}

#[inline(always)]
fn radix3<S: scioxide_core::RealScalar, V: Butterfly<S>, const INVERSE: bool>(
    values: &mut [V; 7],
    roots: &[Complex<S>],
) {
    // With w3 = exp(-2*pi*i/3) = -1/2 - i*sin(pi/3):
    //   X0 = x0 + (x1 + x2)
    //   X1,2 = x0 - (x1 + x2)/2 -/+ i * sin(pi/3) * (x1 - x2)   (forward)
    // roots[1] = w3, so -roots[1].im = sin(pi/3), taken
    // from the plan's root table rather than a scalar constant.
    let half = S::ONE / (S::ONE + S::ONE);
    let root = -roots[1].im;
    let pair = values[1] + values[2];
    let difference = values[1] - values[2];
    let base = values[0] - pair * half;
    let rotated = (difference * root).quarter::<INVERSE>();
    values[0] = values[0] + pair;
    values[1] = base + rotated;
    values[2] = base - rotated;
}

#[inline(always)]
fn radix4<S: scioxide_core::RealScalar, V: Butterfly<S>, const INVERSE: bool>(values: &mut [V; 7]) {
    // Two radix-2 layers; the only non-trivial root is -i (forward) or +i
    // (inverse), applied exactly as a component swap by `quarter`.
    let even_sum = values[0] + values[2];
    let even_difference = values[0] - values[2];
    let odd_sum = values[1] + values[3];
    let odd_difference = values[1] - values[3];
    let rotated = odd_difference.quarter::<INVERSE>();
    values[0] = even_sum + odd_sum;
    values[1] = even_difference + rotated;
    values[2] = even_sum - odd_sum;
    values[3] = even_difference - rotated;
}

#[inline(always)]
fn radix5<S: scioxide_core::RealScalar, V: Butterfly<S>, const INVERSE: bool>(
    values: &mut [V; 7],
    roots: &[Complex<S>],
) {
    // Symmetric radix-5 form: pairing x1/x4 and x2/x3 splits each output into
    // a real-cosine part (shared by conjugate outputs k and 5 - k) and a
    // sine part that only changes sign between them. roots[1] and roots[2]
    // are w5^1 and w5^2.
    let cosine_one = roots[1].re;
    let cosine_two = roots[2].re;
    let sine_one = -roots[1].im;
    let sine_two = -roots[2].im;
    let pair_one = values[1] + values[4];
    let pair_two = values[2] + values[3];
    let difference_one = values[1] - values[4];
    let difference_two = values[2] - values[3];
    let base_one = values[0] + pair_one * cosine_one + pair_two * cosine_two;
    let base_two = values[0] + pair_one * cosine_two + pair_two * cosine_one;
    let rotated_one = (difference_one * sine_one + difference_two * sine_two).quarter::<INVERSE>();
    let rotated_two = (difference_one * sine_two - difference_two * sine_one).quarter::<INVERSE>();
    values[0] = values[0] + pair_one + pair_two;
    values[1] = base_one + rotated_one;
    values[4] = base_one - rotated_one;
    values[2] = base_two + rotated_two;
    values[3] = base_two - rotated_two;
}

#[inline(always)]
fn radix7<S: scioxide_core::RealScalar, V: Butterfly<S>, const INVERSE: bool>(
    values: &mut [V; 7],
    roots: &[Complex<S>],
) {
    // Pair x_r with x_(7-r). The three cosine sums are shared by
    // conjugate output pairs; only the sine rotation changes sign.
    let x0 = values[0];
    let p1 = values[1] + values[6];
    let p2 = values[2] + values[5];
    let p3 = values[3] + values[4];
    let d1 = values[1] - values[6];
    let d2 = values[2] - values[5];
    let d3 = values[3] - values[4];
    let [c1, c2, c3] = [roots[1].re, roots[2].re, roots[3].re];
    let [s1, s2, s3] = [-roots[1].im, -roots[2].im, -roots[3].im];
    let b1 = x0 + p1 * c1 + p2 * c2 + p3 * c3;
    let b2 = x0 + p1 * c2 + p2 * c3 + p3 * c1;
    let b3 = x0 + p1 * c3 + p2 * c1 + p3 * c2;
    let r1 = (d1 * s1 + d2 * s2 + d3 * s3).quarter::<INVERSE>();
    let r2 = (d1 * s2 - d2 * s3 - d3 * s1).quarter::<INVERSE>();
    let r3 = (d1 * s3 - d2 * s1 + d3 * s2).quarter::<INVERSE>();
    values[0] = x0 + p1 + p2 + p3;
    values[1] = b1 + r1;
    values[6] = b1 - r1;
    values[2] = b2 + r2;
    values[5] = b2 - r2;
    values[3] = b3 + r3;
    values[4] = b3 - r3;
}
