//! Fixed-array loops that inline into the caller.
//!
//! `core::array::from_fn` is not always inlined for arrays of large elements,
//! which leaves lane and tangent arithmetic out of line; these loops start from an
//! initialized array instead.

/// Returns `[f(0), …, f(N − 1)]`.
#[inline(always)]
pub(crate) fn build<T: Copy, const N: usize>(fill: T, mut f: impl FnMut(usize) -> T) -> [T; N] {
    let mut out = [fill; N];
    for (index, target) in out.iter_mut().enumerate() {
        *target = f(index);
    }
    out
}

/// Returns `[f(a₀), …]`.
#[inline(always)]
pub(crate) fn map<T: Copy, const N: usize>(a: [T; N], f: impl Fn(T) -> T) -> [T; N] {
    let mut out = a;
    for target in &mut out {
        *target = f(*target);
    }
    out
}

/// Returns `[f(a₀, b₀), …]`.
#[inline(always)]
pub(crate) fn zip<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> T) -> [T; N] {
    let mut out = a;
    for (target, other) in out.iter_mut().zip(b) {
        *target = f(*target, other);
    }
    out
}
