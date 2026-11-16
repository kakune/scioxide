//! Checked floating-point arithmetic: exact sums and products, compensated
//! sums, scaled norms and quadratic roots.
//!
//! Exact reductions store signed integer significands in a fixed-size
//! superaccumulator and round once when a scalar result is requested. Pair and
//! triple products are split into exact integer significands, so a product that
//! would overflow or underflow on its own does not affect the sum.
//!
//! Exact sums, products, sums of pair and triple products, product differences,
//! midpoints and affine combinations are correctly rounded (0 ulps). An exact
//! zero is `+0`; a nonzero negative value that rounds to zero is `-0`. Endpoint
//! returns, `next_after` and power-of-two scaling keep the zero signs their docs
//! state. Quotients and accumulated quotients round once. Product quotients are
//! within 4 ulps, including subnormal results. LASSQ norms have an
//! order-dependent `O(n ε)` error bound. Quadratic roots have a scaled-residual
//! bound, because their forward condition number is unbounded.
//!
//! Compensated sums ([`CompensatedSum`], [`PromotingCompensatedSum`] and the
//! short lane reductions) work in the scalar's precision. They keep first-order
//! rounding residuals but give neither an exact sign nor a correctly rounded sum.

mod accumulator;
mod checked;
mod commit;
mod compensated;
mod fast;
mod lanes;
mod norm;
mod product_reduce;
mod product_sum;
mod promoting;
mod quadratic;
mod real_product_sum;
mod real_value_sum;
mod short;

pub use accumulator::RangeAccumulator;
#[cfg(all(
    not(feature = "std"),
    not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
pub(crate) use accumulator::finite_fused_mul_add;
pub use checked::{
    absolute_difference_le, checked_affine, checked_affine_ratio, checked_difference,
    checked_difference_of_products, checked_float_product, checked_float_product_factors,
    checked_float_quotient, checked_midpoint, checked_mul_add, checked_next_after,
    checked_product_quotient, checked_scalbn, checked_sum, checked_sum_of_products, opposite_signs,
};
pub use commit::{checked_final_rounding, validate_then_commit};
pub use compensated::{
    CompensatedSum, compensated_indexed, compensated_reduce, finish_compensated_lanes,
};
pub use fast::{HARDWARE_FMA, fast_mul_add};
pub use lanes::{ShortReductionRange, four_lane_sums};
pub use norm::{ScaledNorm, checked_scaled_norm};
pub use product_reduce::checked_product_sum;
pub use product_sum::ProductSum;
pub use promoting::{PromotingCompensatedSum, PromotingProduct};
pub use quadratic::checked_quadratic_roots;
pub use real_product_sum::{ProductFactor, RealProductSum};
pub use real_value_sum::RealValueSum;
pub use short::{
    certified_short_sum_products, certified_short_sum_triples, certified_sum_two_products,
    certified_sum_two_products_with, two_product,
};

use crate::{NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError};

/// Returns the rounded sum `a + b` and its rounding error (Knuth's TwoSum).
///
/// If `a`, `b` and the intermediate sums and differences are finite,
/// `sum + residual` equals `a + b` exactly, including under gradual underflow.
/// The function does not check its inputs: the caller must ensure finiteness or
/// reject a non-finite result.
///
/// ```
/// let (sum, residual) = scioxide_core::backend::two_sum(1e16_f64, 1.0);
/// assert_eq!(sum, 1e16);
/// assert_eq!(residual, 1.0);
/// ```
#[inline]
pub fn two_sum<S: RealScalar>(a: S, b: S) -> (S, S) {
    let sum = a + b;
    let virtual_b = sum - a;
    (sum, (a - (sum - virtual_b)) + (b - virtual_b))
}

/// Certify an enclosure `rounded + tail ± bound` strictly inside one rounding
/// cell. Callers supply a nonnegative upper bound on the residual error.
/// Outward-rounded endpoints account for their own rounding; separate exact
/// half-spacings keep the asymmetry at signed powers of two. Unrepresentable
/// half-spacings, midpoint ties and uncertain enclosures request exact fallback.
#[inline]
fn certified_rounding_cell<S: RealScalar>(rounded: S, tail: S, bound: S) -> Option<S> {
    if !rounded.is_normal() || rounded.abs() == S::MAX {
        return None;
    }
    let low = (tail - bound).next_after(-S::MAX);
    let high = (tail + bound).next_after(S::MAX);
    let below = (rounded - rounded.next_after(-S::MAX)) * S::HALF;
    let above = (rounded.next_after(S::MAX) - rounded) * S::HALF;
    (low > -below && high < above).then_some(rounded)
}

fn overflow(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Overflow, detail)
}

fn non_finite_input(detail: &'static str) -> SciOxideError {
    SciOxideError::non_finite(NonFiniteOrigin::Input, detail)
}

fn validate_finite<S: RealScalar>(value: S, detail: &'static str) -> Result<(), SciOxideError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(non_finite_input(detail))
    }
}
