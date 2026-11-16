//! Euler–Maclaurin tail correction with a conditional analytic remainder bound.

use super::runner::charge;
use super::workspace::SeriesScratch;
use super::{EulerMaclaurinProvider, SeriesError, SeriesState, bound, checked_callback_number};
use scioxide_core::{
    AlgebraicScalar, EvaluationRole, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

// Exact Bernoulli rationals, rounded once as in the existing Hurwitz-zeta
// implementation. Bounds use the next larger representable magnitude.
const BERNOULLI_OVER_FACTORIAL: [f64; 12] = [
    f64::from_bits(0x3fb5_5555_5555_5555),
    f64::from_bits(0xbf56_c16c_16c1_6c17),
    f64::from_bits(0x3f01_566a_bc01_1567),
    f64::from_bits(0xbeab_bd77_9334_ef0b),
    f64::from_bits(0x3e56_6a8f_2bf7_0ebe),
    f64::from_bits(0xbe02_2805_d644_267f),
    f64::from_bits(0x3dad_6db2_c4e0_9162),
    f64::from_bits(0xbd57_da4e_1f79_955c),
    f64::from_bits(0x3d03_5587_1d65_2e9e),
    f64::from_bits(0xbcaf_57d9_68ca_acf1),
    f64::from_bits(0x3c59_67e1_f09c_376f),
    f64::from_bits(0xbc04_97d9_033a_2b5c),
];

/// Largest supported number of Euler–Maclaurin corrections.
pub(super) const MAX_ORDER: usize = BERNOULLI_OVER_FACTORIAL.len();

/// Stage the Euler–Maclaurin estimate for the new row `state.count` at series
/// index `index`, returning the provider calls made.
///
/// Euler-Maclaurin tail from index n:
///     sum_{k>=n} f(k) = int_n^inf f + f(n)/2 - sum_{j=1}^{p} B_2j/(2j)! f^(2j-1)(n) + R_p,
///     |R_p| <= |B_2p|/(2p)! int_n^inf |f^(2p)|,
/// so the estimate is the partial sum before n plus these corrections. Provider errors
/// and the coefficients are rounded upward (`next_after`) so the reported bound is
/// not weakened by rounding. Output buffers are NaN-poisoned before every callback.
/// Only the previous partial sum and the new term row are read. The arithmetic floor
/// is `64 * epsilon` times each addend of the floating-point estimate, the correctly
/// rounded prefix included, accumulated without overflow.
pub(super) fn candidate<T: NumericScalar>(
    provider: &dyn EulerMaclaurinProvider<T>,
    index: usize,
    state: &SeriesState<T>,
    scratch: &mut SeriesScratch<T>,
    context: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let width = state.width;
    let row = state.offset(state.count);
    let previous = (state.count > 0).then(|| state.offset(state.count - 1));
    let floor = T::Real::try_from_f64_lossy(64.0)? * T::Real::EPSILON;
    scratch
        .derivative
        .fill(T::from_components(T::Real::NAN, T::Real::NAN));
    scratch.provider_error.fill(T::Real::NAN);
    charge(context, EvaluationRole::Function)?;
    provider.tail_integral(index, &mut scratch.derivative, &mut scratch.provider_error)?;
    for c in 0..width {
        let prefix = previous.map_or(T::ZERO, |offset| state.partials[offset + c]);
        let term = state.terms[row + c];
        let integral = checked_callback_number(scratch.derivative[c])?;
        let half = T::from_real(T::Real::HALF) * term;
        scratch.staged[c].value = checked_callback_number(prefix + integral + half)?;
        scratch.errors[c] = SeriesError {
            provider: Some(bound(scratch.provider_error[c])?),
            roundoff: floor * prefix.abs() + floor * integral.abs() + floor * half.abs(),
            ..SeriesError::default()
        };
    }
    for (j, &coefficient) in BERNOULLI_OVER_FACTORIAL[..state.order].iter().enumerate() {
        let coefficient = T::Real::try_from_f64_lossy(coefficient)?;
        let upper = coefficient.abs().next_after(T::Real::INFINITY);
        scratch
            .derivative
            .fill(T::from_components(T::Real::NAN, T::Real::NAN));
        scratch.provider_error.fill(T::Real::NAN);
        charge(context, EvaluationRole::Gradient)?;
        provider.odd_derivative(
            index,
            2 * j + 1,
            &mut scratch.derivative,
            &mut scratch.provider_error,
        )?;
        for c in 0..width {
            let correction =
                T::from_real(coefficient) * checked_callback_number(scratch.derivative[c])?;
            scratch.staged[c].value =
                checked_callback_number(scratch.staged[c].value - correction)?;
            scratch.errors[c].roundoff += floor * correction.abs();
            let error = scratch.errors[c].provider.unwrap_or(T::Real::ZERO)
                + upper * bound(scratch.provider_error[c])?;
            scratch.errors[c].provider = Some(bound(error.next_after(T::Real::INFINITY))?);
        }
    }
    scratch.l1_bound.fill(T::Real::NAN);
    charge(context, EvaluationRole::Gradient)?;
    provider.even_derivative_l1_bound(index, 2 * state.order, &mut scratch.l1_bound)?;
    let coefficient = T::Real::try_from_f64_lossy(BERNOULLI_OVER_FACTORIAL[state.order - 1].abs())?
        .next_after(T::Real::INFINITY);
    for c in 0..width {
        let tail = coefficient * bound(scratch.l1_bound[c])?;
        scratch.errors[c].tail_bound = Some(bound(tail.next_after(T::Real::INFINITY))?);
    }
    Ok(state.order + 2)
}
