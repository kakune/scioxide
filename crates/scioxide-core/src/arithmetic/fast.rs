//! Multiply-add in the fastest inline form for the target.

use crate::RealScalar;

/// Whether [`RealScalar::mul_add`] compiles to one inline fused instruction on
/// this target: the x86 `fma` or AArch64 `neon` target feature is enabled at
/// compile time.
///
/// On other targets `mul_add` is still correctly rounded but runs out of line
/// (a library call with `std`, software emulation without it), which can
/// dominate the cost of short polynomial and recurrence loops.
///
/// # Examples
/// ```
/// let fused = cfg!(target_arch = "aarch64")
///     || cfg!(all(target_arch = "x86_64", target_feature = "fma"));
/// assert!(!fused || scioxide_core::HARDWARE_FMA);
/// ```
pub const HARDWARE_FMA: bool = cfg!(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
));

/// Computes `left · right + addend` in the cheapest inline form for the target.
///
/// If [`HARDWARE_FMA`] is true, this is the fused operation: it rounds once and
/// equals [`RealScalar::mul_add`] bit for bit. Otherwise it multiplies and adds
/// separately, rounding twice, and never calls a library or software FMA. Use
/// it only where a second rounding is acceptable, such as Horner steps of an
/// approximation whose error bound allows it. Error-free transformations need
/// [`RealScalar::mul_add`] or a split product instead.
///
/// # Examples
/// ```
/// use scioxide_core::fast_mul_add;
///
/// assert_eq!(fast_mul_add(2.0_f64, 3.0, 1.0), 7.0);
/// assert_eq!(fast_mul_add(0.5_f32, 4.0, -2.0), 0.0);
/// ```
#[inline(always)]
pub fn fast_mul_add<S: RealScalar>(left: S, right: S, addend: S) -> S {
    if HARDWARE_FMA {
        left.mul_add(right, addend)
    } else {
        left * right + addend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_follows_the_target() {
        // (1 + eps)(1 - eps) - 1 = -eps^2 is lost by the separate product.
        let epsilon = f64::EPSILON;
        let value = fast_mul_add(1.0 + epsilon, 1.0 - epsilon, -1.0);
        if HARDWARE_FMA {
            assert_eq!(value, -epsilon * epsilon);
        } else {
            assert_eq!(value, 0.0);
        }
        assert_eq!(fast_mul_add(3.0_f64, 0.25, 0.5), 1.25);
        assert!(fast_mul_add(f64::INFINITY, 0.0, 1.0).is_nan());
    }
}
