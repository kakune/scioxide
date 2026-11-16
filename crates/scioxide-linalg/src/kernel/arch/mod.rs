//! Architecture-specific packed matrix kernel leaves.

use scioxide_core::NumericScalar;
use scioxide_core::backend::KernelDispatch;

use super::microkernel::Microkernel;

#[allow(
    unsafe_code,
    reason = "verified SIMD leaves operate on checked private vector borrows"
)]
pub(crate) mod recurrence;

#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "reviewed ISA leaves consume verified dispatch and validated borrows"
)]
mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(
    unsafe_code,
    reason = "reviewed ISA leaves consume verified dispatch and validated borrows"
)]
mod x86;

/// The real leaf evaluating packed `S` products. Complex scalars use the leaf of
/// their component type on component-layout panels (the 1m formulation), so
/// a complex register tile is `mr x nr/2` with a real depth of twice the shared depth.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    expect(
        clippy::extra_unused_type_parameters,
        reason = "typed dispatch retains its scalar bound on targets without architecture leaves"
    )
)]
pub(crate) fn select_microkernel<S: NumericScalar>(
    _dispatch: KernelDispatch,
) -> Option<Microkernel> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(kernel) = x86::select_microkernel::<S::Real>(_dispatch) {
        return Some(kernel);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(kernel) = aarch64::select_microkernel::<S::Real>(_dispatch) {
        return Some(kernel);
    }
    None
}

/// Contiguous real dot candidate; the token and slice lengths prove leaf safety.
/// Callers must check finiteness and restore or retry the range frame before publication.
pub(crate) fn dot<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<Option<S>, scioxide_core::SciOxideError> {
    if left.len() != right.len() {
        return Err(scioxide_core::SciOxideError::dimension_mismatch(
            [left.len()],
            [right.len()],
        ));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::dot(left, right, dispatch)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::dot(left, right, dispatch)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (left, right, dispatch);
        Ok(None)
    }
}

/// Fused finite validation and component bound for a contiguous real operand.
pub(crate) fn maximum<S: NumericScalar>(
    values: &[S],
    dispatch: KernelDispatch,
) -> Result<Option<S::Real>, scioxide_core::SciOxideError> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::maximum(values, dispatch)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::maximum(values, dispatch)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (values, dispatch);
        Ok(None)
    }
}

/// Bounded affine update: validation proves every lane is representable.
pub(crate) fn affine<S: NumericScalar>(
    alpha: S,
    left: &[S],
    beta: S,
    right: &[S],
    output: &mut [S],
    dispatch: KernelDispatch,
) -> bool {
    if left.len() != right.len() || output.len() != left.len() {
        return false;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::affine(alpha, left, beta, right, output, dispatch)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::affine(alpha, left, beta, right, output, dispatch)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (alpha, left, beta, right, output, dispatch);
        false
    }
}

/// Optional verified x86 acceleration of the safe contiguous GKO kernel.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    expect(unused_variables, reason = "GKO uses the portable kernel outside x86")
)]
pub(crate) fn gko_subtract_product<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: scioxide_core::Complex<T::Real>,
    dispatch: KernelDispatch,
) -> Option<bool> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::gko_subtract_product(target, start, source, source_start, count, factor, dispatch)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        None
    }
}

/// Optional verified x86 acceleration of the safe contiguous GKO kernel.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    expect(unused_variables, reason = "GKO uses the portable kernel outside x86")
)]
pub(crate) fn gko_reconstruct_line<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (
        scioxide_core::Complex<T::Real>,
        scioxide_core::Complex<T::Real>,
    ),
    node: scioxide_core::Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
    dispatch: KernelDispatch,
) -> Option<bool> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::gko_reconstruct_line(sources, weights, node, range, output, offset, dispatch)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        None
    }
}

#[cfg(all(
    test,
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::aarch64::dot as leaf_dot;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::x86::dot as leaf_dot;
    use super::*;
    use scioxide_core::backend::KernelBackend;

    #[test]
    fn safe_dot_leaves_check_lengths_and_handle_unaligned_tails() {
        let processor = crate::kernel::tests::test_processor();
        for backend in KernelBackend::available(processor) {
            let dispatch = KernelDispatch::for_backend(backend, processor).unwrap();
            for length in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
                let a = [0.5_f64; 67];
                let b = [2.0_f64; 67];
                let c = [0.5_f32; 67];
                let d = [2.0_f32; 67];
                for (left, right) in [
                    (&a[1..1 + length], &b[1..2 + length]),
                    (&a[1..2 + length], &b[1..1 + length]),
                ] {
                    assert_eq!(
                        leaf_dot(left, right, dispatch).unwrap_err().kind(),
                        scioxide_core::SciOxideErrorKind::DimensionMismatch
                    );
                    assert_eq!(
                        dot(left, right, dispatch).unwrap_err().kind(),
                        scioxide_core::SciOxideErrorKind::DimensionMismatch
                    );
                }
                for (left, right) in [
                    (&c[1..1 + length], &d[1..2 + length]),
                    (&c[1..2 + length], &d[1..1 + length]),
                ] {
                    assert_eq!(
                        leaf_dot(left, right, dispatch).unwrap_err().kind(),
                        scioxide_core::SciOxideErrorKind::DimensionMismatch
                    );
                }
                let actual = leaf_dot(&a[1..1 + length], &b[1..1 + length], dispatch).unwrap();
                let actual32 = leaf_dot(&c[1..1 + length], &d[1..1 + length], dispatch).unwrap();
                if backend == KernelBackend::Scalar {
                    assert!(actual.is_none() && actual32.is_none());
                } else {
                    assert!((actual.unwrap() - length as f64).abs() < 1e-14);
                    assert!((actual32.unwrap() - length as f32).abs() < 1e-6);
                }
            }
        }
    }
}
