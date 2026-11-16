//! Architecture-specific FFT codelet selection.

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use scioxide_core::backend::KernelBackend;
use scioxide_core::backend::KernelDispatch;

use super::StageKernels;

#[expect(
    unsafe_code,
    reason = "reviewed dispatch admits borrows against immutable plan proofs"
)]
mod dispatch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[expect(
    unsafe_code,
    reason = "shared register arithmetic is inlined into reviewed ISA leaves"
)]
mod codelets;

// The caller's unsafe boundary proves host features and whole radix*width
// blocks with (radix-1)*width twiddles. The vector loop never crosses a
// branch; the same scalar tail handles every register width.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! radix_loop {
    ($scalar:ty, $inverse:ident, $data:ident, $radix:expr, $width:ident,
     $twiddles:ident, $roots:ident, $lanes:expr, $zero:expr,
     $load:expr, $store:expr, $mul:expr, |$values:ident| $codelet:expr) => {{
        let vector_width = $width / $lanes * $lanes;
        let block_size = $radix * $width;
        let mut block_start = 0;
        while block_start < $data.len() {
            let mut index = 0;
            while index < vector_width {
                let mut $values = [$zero; 7];
                $values[0] = $load($data, block_start + index);
                let mut branch = 1;
                while branch < $radix {
                    $values[branch] = $mul(
                        $load($data, block_start + branch * $width + index),
                        $load($twiddles, (branch - 1) * $width + index),
                    );
                    branch += 1;
                }
                $codelet;
                branch = 0;
                while branch < $radix {
                    $store(
                        $data,
                        block_start + branch * $width + index,
                        $values[branch],
                    );
                    branch += 1;
                }
                index += $lanes;
            }
            let block = &mut $data[block_start..block_start + block_size];
            for tail in vector_width..$width {
                match $radix {
                    2 => scalar::butterfly::<$scalar, 2, $inverse>(
                        block, $width, tail, $twiddles, $roots,
                    ),
                    3 => scalar::butterfly::<$scalar, 3, $inverse>(
                        block, $width, tail, $twiddles, $roots,
                    ),
                    4 => scalar::butterfly::<$scalar, 4, $inverse>(
                        block, $width, tail, $twiddles, $roots,
                    ),
                    5 => scalar::butterfly::<$scalar, 5, $inverse>(
                        block, $width, tail, $twiddles, $roots,
                    ),
                    7 => scalar::butterfly::<$scalar, 7, $inverse>(
                        block, $width, tail, $twiddles, $roots,
                    ),
                    _ => unreachable!(),
                }
            }
            block_start += block_size;
        }
    }};
}

#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "reviewed ISA leaves consume verified dispatch and validated borrows"
)]
mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[expect(
    unsafe_code,
    reason = "reviewed ISA leaves consume verified dispatch and validated borrows"
)]
mod x86;

pub(crate) fn kernels_f32(
    dispatch: KernelDispatch,
    radix: usize,
    width: usize,
) -> StageKernels<f32> {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = (radix, width);

    // The token proves host ISA support independently of provider claims and
    // is matched only here. FFT leaves exist for AVX2/FMA and NEON; any other
    // token uses the complete scalar codelets (preparation rejects a required
    // AVX-512 backend before a token is minted).
    match dispatch.backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelBackend::Avx2Fma => x86::avx2_kernels_f32(radix, width),
        #[cfg(target_arch = "aarch64")]
        KernelBackend::Neon => aarch64::kernels_f32(),
        _ => super::scalar::kernels(),
    }
}

pub(crate) fn kernels_f64(
    dispatch: KernelDispatch,
    radix: usize,
    width: usize,
) -> StageKernels<f64> {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = (radix, width);

    match dispatch.backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelBackend::Avx2Fma => x86::avx2_kernels_f64(radix, width),
        #[cfg(target_arch = "aarch64")]
        KernelBackend::Neon => aarch64::kernels_f64(),
        _ => super::scalar::kernels(),
    }
}

/// Run a lane-block pass: an AVX2/FMA build of its portable body on hosts whose
/// verified dispatch provides it, the portable body otherwise. Rust never
/// contracts multiplies and additions, so both builds give identical bits.
#[inline]
pub(crate) fn run_pass<P: super::block::BlockPass>(dispatch: KernelDispatch, pass: P) -> P::Output {
    match dispatch.backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelBackend::Avx2Fma => x86::avx2_pass(dispatch, pass),
        _ => pass.run(),
    }
}

/// Add one weighted NUFFT row with Kahan compensation.
pub(crate) fn nufft_spread<S: scioxide_core::RealScalar>(
    dispatch: KernelDispatch,
    output: &mut [scioxide_core::Complex<S>],
    correction: &mut [scioxide_core::Complex<S>],
    weights: &[scioxide_core::Complex<S>],
    value: scioxide_core::Complex<S>,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if dispatch.backend() == KernelBackend::Avx2Fma {
        use scioxide_core::backend::{ComplexSlice, ComplexSliceMut};
        match (
            S::complex_slice_mut(output),
            S::complex_slice_mut(correction),
            S::complex_slice(weights),
            S::complex_slice(core::slice::from_ref(&value)),
        ) {
            (
                ComplexSliceMut::F32(out),
                ComplexSliceMut::F32(c),
                ComplexSlice::F32(w),
                ComplexSlice::F32(v),
            ) => x86::nufft_spread_f32(dispatch, out, c, w, v[0]),
            (
                ComplexSliceMut::F64(out),
                ComplexSliceMut::F64(c),
                ComplexSlice::F64(w),
                ComplexSlice::F64(v),
            ) => x86::nufft_spread_f64(dispatch, out, c, w, v[0]),
            _ => unreachable!("one scalar representation for all NUFFT rows"),
        }
        return;
    }
    let _ = dispatch;
    for ((out, correction), weight) in output.iter_mut().zip(correction).zip(weights) {
        let adjusted = value * weight.re - *correction;
        let next = *out + adjusted;
        *correction = (next - *out) - adjusted;
        *out = next;
    }
}

/// Native three-dimensional interpolation, when supported by this backend.
pub(crate) fn nufft_interpolate3<S: scioxide_core::RealScalar>(
    dispatch: KernelDispatch,
    grid: &[scioxide_core::Complex<S>],
    weights: &[scioxide_core::Complex<S>],
    axes: [(usize, usize, usize); 3],
    width: usize,
) -> Option<scioxide_core::Complex<S>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if dispatch.backend() == KernelBackend::Avx2Fma {
        use scioxide_core::backend::{ComplexSlice, ComplexSliceMut};
        let mut result = scioxide_core::Complex::new(S::ZERO, S::ZERO);
        if let (ComplexSlice::F64(grid), ComplexSlice::F64(weights), ComplexSliceMut::F64(output)) = (
            S::complex_slice(grid),
            S::complex_slice(weights),
            S::complex_slice_mut(core::slice::from_mut(&mut result)),
        ) {
            output[0] = x86::nufft_interpolate3_f64(dispatch, grid, weights, axes, width);
            return Some(result);
        }
    }
    let _ = (dispatch, grid, weights, axes, width);
    None
}

/// Spread one point into a bounded local three-dimensional tile.
pub(crate) fn nufft_tile<S: scioxide_core::RealScalar>(
    dispatch: KernelDispatch,
    tile: &mut [scioxide_core::Complex<S>],
    weights: &[scioxide_core::Complex<S>],
    value: scioxide_core::Complex<S>,
    side: usize,
    axes: [(usize, usize); 3],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if dispatch.backend() == KernelBackend::Avx2Fma {
        use scioxide_core::backend::{ComplexSlice, ComplexSliceMut};
        match (
            S::complex_slice_mut(tile),
            S::complex_slice(weights),
            S::complex_slice(core::slice::from_ref(&value)),
        ) {
            (ComplexSliceMut::F32(tile), ComplexSlice::F32(w), ComplexSlice::F32(v)) => {
                x86::nufft_tile_f32(dispatch, tile, w, v[0], side, axes)
            }
            (ComplexSliceMut::F64(tile), ComplexSlice::F64(w), ComplexSlice::F64(v)) => {
                x86::nufft_tile_f64(dispatch, tile, w, v[0], side, axes)
            }
            _ => unreachable!("one scalar representation for all NUFFT rows"),
        }
        return;
    }
    let _ = dispatch;
    let [(x, nx), (y, ny), (z, nz)] = axes;
    let width = weights.len() / 3;
    for (i, wx) in weights[..nx].iter().enumerate() {
        for (j, wy) in weights[width..width + ny].iter().enumerate() {
            let value = value * (wx.re * wy.re);
            let start = ((x + i) * side + y + j) * side + z;
            for (out, weight) in tile[start..start + nz]
                .iter_mut()
                .zip(&weights[2 * width..2 * width + nz])
            {
                *out += value * weight.re;
            }
        }
    }
}
