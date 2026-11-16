//! Independent defining sums and explicit numerical comparison bounds.
//!
//! These helpers use binary64 arithmetic, never production FFT twiddles. Small
//! defining sums are structural oracles; generated high-precision fixtures own
//! the tight forward-error envelope. Callers supply their mathematical error
//! metric and bound: component error, complex norm, relative L2, or scaled error.
//! Backend sweeps iterate only the host-verified CPU backends that have FFT leaves.
#![allow(
    dead_code,
    reason = "each test target uses only part of this shared support"
)]
#![allow(
    clippy::unwrap_used,
    reason = "test inputs use the built-in f32/f64 scalars"
)]

use scioxide_core::backend::KernelBackend;
use scioxide_core::{Complex, ProcessorCapabilities, RealScalar};
use std::{fmt::Display, vec::Vec};

/// Round a binary64 test constant to `S`.
pub(crate) fn scalar<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

/// Widen an `S` value to binary64, exactly.
pub(crate) fn wide<S: RealScalar>(value: S) -> f64 {
    value.try_to_f64_exact().unwrap()
}

/// Scalar epsilon of `S` in binary64.
pub(crate) fn epsilon<S: RealScalar>() -> f64 {
    wide(S::EPSILON)
}

/// Widen both complex components to binary64 before arithmetic.
pub(crate) fn widen<S: RealScalar>(value: Complex<S>) -> Complex<f64> {
    Complex::new(wide(value.re), wide(value.im))
}

/// Hermitian inner product, accumulated in binary64 on equal-length inputs.
pub(crate) fn dot<S: RealScalar>(a: &[Complex<S>], b: &[Complex<S>]) -> Complex<f64> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).fold(Complex::new(0.0, 0.0), |sum, (a, b)| {
        sum + widen(*a).conj() * widen(*b)
    })
}

/// Euclidean norm with binary64 accumulation.
pub(crate) fn norm<S: RealScalar>(values: &[Complex<S>]) -> f64 {
    dot(values, values).re.sqrt()
}

/// One unnormalized Fourier output, with `(phase in radians, real weight)` per
/// input. Zero weights exclude terms (for example, other N-D batches). No
/// allocations occur, including when called by an injected test provider.
pub(crate) fn direct_sum<S: RealScalar>(
    input: &[Complex<S>],
    kernel: impl Fn(usize) -> (f64, f64),
) -> Complex<f64> {
    input
        .iter()
        .enumerate()
        .fold(Complex::new(0.0, 0.0), |sum, (j, x)| {
            let (phase, weight) = kernel(j);
            if weight == 0.0 {
                return sum;
            }
            let (sin, cos) = phase.sin_cos();
            sum + widen(*x) * Complex::new(cos, sin) * weight
        })
}

/// Ordinary unnormalized forward (negative sign) or inverse (positive sign) DFT.
pub(crate) fn direct_dft(input: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..input.len())
        .map(|k| {
            direct_sum(input, |j| {
                (
                    sign * core::f64::consts::TAU * (j * k % input.len()) as f64
                        / input.len() as f64,
                    1.0,
                )
            })
        })
        .collect()
}

/// The single comparison rule: a finite, nonnegative error must be at most a
/// finite, nonnegative bound. Zero permits exact equality. Callers derive bounds
/// from scalar epsilon, transform length, reference scale, or a fixture's
/// documented absolute allowance; this function adds no hidden tolerance.
#[track_caller]
pub(crate) fn assert_close(error: f64, bound: f64, context: impl Display) {
    assert!(
        error.is_finite() && error >= 0.0 && bound.is_finite() && bound >= 0.0 && error <= bound,
        "{context}: error={error:e}, bound={bound:e}"
    );
}

/// Maximum absolute component, for componentwise complex comparisons.
pub(crate) fn component_max(value: Complex<f64>) -> f64 {
    value.re.abs().max(value.im.abs())
}

/// Compare equal-length complex slices with an explicit error norm and a bound
/// computed from each reference value. Global reference scales can be captured
/// by the bound closure. Conversion precedes subtraction to retain f32 error.
#[track_caller]
pub(crate) fn assert_complex_close<S: RealScalar, T: RealScalar>(
    actual: &[Complex<S>],
    expected: &[Complex<T>],
    norm: impl Fn(Complex<f64>) -> f64,
    bound: impl Fn(Complex<f64>) -> f64,
) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (widen(*a), widen(*b));
        assert!(
            a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
            "index {index}: {a:?} != {b:?}"
        );
        assert_close(
            norm(a - b),
            bound(b),
            format_args!("index {index}: {a:?} != {b:?}"),
        );
    }
}

/// Compare relative L2 error with an explicit relative allowance. An all-zero
/// reference requires exact zero output; non-finite values always fail.
#[track_caller]
pub(crate) fn assert_complex_l2_close<S: RealScalar, T: RealScalar>(
    actual: &[Complex<S>],
    expected: &[Complex<T>],
    relative_bound: f64,
) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    assert!(relative_bound.is_finite() && relative_bound >= 0.0);
    let error = actual
        .iter()
        .zip(expected)
        .fold(0.0, |sum, (a, b)| sum + (widen(*a) - widen(*b)).norm_sqr())
        .sqrt();
    let scale = norm(expected);
    assert_close(error, relative_bound * scale, "complex L2 error");
}

/// The FFT componentwise rule: every component error is at most
/// `256 epsilon(S) max(1, log2(work))` times the largest expected component, or
/// one if that is smaller. `work` is the transform length, or the product of
/// the transformed extents of an N-D transform.
#[track_caller]
pub(crate) fn assert_fft_close<S: RealScalar, T: RealScalar>(
    actual: &[Complex<S>],
    expected: &[Complex<T>],
    work: usize,
) {
    let magnitude = expected
        .iter()
        .map(|&value| component_max(widen(value)))
        .fold(1.0, f64::max);
    let bound = 256.0 * epsilon::<S>() * (work as f64).log2().max(1.0) * magnitude;
    assert_complex_close(actual, expected, component_max, |_| bound);
}

/// Absolute real error, with the same length and finiteness rules as complex
/// comparisons. Relative allowances are explicit in the reference-bound closure.
#[track_caller]
pub(crate) fn assert_real_close<S: RealScalar, T: RealScalar>(
    actual: &[S],
    expected: &[T],
    bound: impl Fn(f64) -> f64,
) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (wide(*a), wide(*b));
        assert_close(
            (a - b).abs(),
            bound(b),
            format_args!("index {index}: {a:e} != {b:e}"),
        );
    }
}

/// Host-verified CPU backends that have FFT leaves. AVX-512 has none, and
/// requiring it is rejected at preparation on every host.
pub(crate) fn fft_backends(
    processor: &dyn ProcessorCapabilities,
) -> impl Iterator<Item = KernelBackend> + '_ {
    KernelBackend::available(processor).filter(|&backend| backend != KernelBackend::Avx512)
}
