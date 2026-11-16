//! Fixed-storage phase reduction for oscillatory special functions.

use crate::support::wide::Wide;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // std-linked tests can supply the inherent float methods.
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

const SQRT_FRACTION_BITS: usize = 1_664;
const RECIPROCAL_FRACTION_BITS: usize = 2_048;
const SQRT_LIMBS: usize = 28;
const MANTISSA_PRODUCT_LIMBS: usize = SQRT_LIMBS + 1;
const RECIPROCAL_LIMBS: usize = 32;
const PRODUCT_LIMBS: usize = MANTISSA_PRODUCT_LIMBS + RECIPROCAL_LIMBS;

use super::phase_constants::RECIPROCAL_THREE_PI;

fn shift_left(value: &mut [u64], amount: u32) {
    debug_assert!(amount < 64);
    let mut carry = 0_u64;
    value.iter_mut().for_each(|limb| {
        let next_carry = if amount == 0 {
            0
        } else {
            *limb >> (64 - amount)
        };
        *limb = (*limb << amount) | carry;
        carry = next_carry;
    });
}

fn greater_or_equal(left: &[u64], right: &[u64]) -> bool {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .find_map(|(&lhs, &rhs)| (lhs != rhs).then_some(lhs > rhs))
        .unwrap_or(true)
}

fn subtract(left: &mut [u64], right: &[u64]) {
    let mut borrow = 0_u64;
    left.iter_mut().zip(right).for_each(|(lhs, &rhs)| {
        let (without_rhs, first_borrow) = lhs.overflowing_sub(rhs);
        let (result, second_borrow) = without_rhs.overflowing_sub(borrow);
        *lhs = result;
        borrow = u64::from(first_borrow || second_borrow);
    });
    debug_assert_eq!(borrow, 0);
}

fn shifted_radicand_bit(value: u64, position: usize) -> u64 {
    position
        .checked_sub(2 * SQRT_FRACTION_BITS)
        .filter(|&bit| bit < u64::BITS as usize)
        .map_or(0, |bit| (value >> bit) & 1)
}

/// Compute `floor(sqrt(value) * 2^SQRT_FRACTION_BITS)` with restoring binary
/// square root. Storage and work are fixed for the complete IEEE exponent range.
fn fixed_sqrt(value: u64) -> [u64; SQRT_LIMBS] {
    let value_bits = u64::BITS as usize - value.leading_zeros() as usize;
    let radicand_bits = 2 * SQRT_FRACTION_BITS + value_bits;
    let pairs = radicand_bits.div_ceil(2);
    let mut root = [0_u64; SQRT_LIMBS];
    let mut remainder = [0_u64; SQRT_LIMBS];

    (0..pairs).rev().for_each(|pair_index| {
        shift_left(&mut remainder, 2);
        let low = shifted_radicand_bit(value, 2 * pair_index);
        let high = shifted_radicand_bit(value, 2 * pair_index + 1);
        remainder[0] |= low | (high << 1);

        let mut trial = root;
        shift_left(&mut trial, 2);
        trial[0] |= 1;
        shift_left(&mut root, 1);
        if greater_or_equal(&remainder, &trial) {
            subtract(&mut remainder, &trial);
            root[0] |= 1;
        }
    });
    root
}

fn multiply_mantissa(value: &[u64; SQRT_LIMBS], mantissa: u64) -> [u64; MANTISSA_PRODUCT_LIMBS] {
    let mut product = [0_u64; MANTISSA_PRODUCT_LIMBS];
    let mut carry = 0_u128;
    value.iter().enumerate().for_each(|(index, &limb)| {
        let accumulated = u128::from(limb) * u128::from(mantissa) + carry;
        product[index] = accumulated as u64;
        carry = accumulated >> 64;
    });
    product[SQRT_LIMBS] = carry as u64;
    product
}

fn multiply_reciprocal(value: &[u64; MANTISSA_PRODUCT_LIMBS]) -> [u64; PRODUCT_LIMBS] {
    let mut product = [0_u64; PRODUCT_LIMBS];
    value.iter().enumerate().for_each(|(left_index, &left)| {
        let mut carry = 0_u128;
        RECIPROCAL_THREE_PI
            .iter()
            .enumerate()
            .for_each(|(right_index, &right)| {
                let output_index = left_index + right_index;
                let accumulated = u128::from(left) * u128::from(right)
                    + u128::from(product[output_index])
                    + carry;
                product[output_index] = accumulated as u64;
                carry = accumulated >> 64;
            });
        product[left_index + RECIPROCAL_LIMBS] = carry as u64;
    });
    product
}

fn bit(value: &[u64; PRODUCT_LIMBS], position: i64) -> u64 {
    let Ok(position) = usize::try_from(position) else {
        return 0;
    };
    value
        .get(position / 64)
        .map_or(0, |limb| (limb >> (position % 64)) & 1)
}

fn leading_fraction(value: &[u64; PRODUCT_LIMBS], binary_point: i64) -> (f64, f64) {
    let bits = (0..128_u32).fold(0_u128, |result, offset| {
        (result << 1) | u128::from(bit(value, binary_point - i64::from(offset) - 1))
    });
    [
        ((bits >> 96) as u32, -32),
        ((bits >> 64) as u32, -64),
        ((bits >> 32) as u32, -96),
        (bits as u32, -128),
    ]
    .into_iter()
    .fold((0.0, 0.0), |sum, (chunk, exponent)| {
        let value = Wide {
            hi: sum.0,
            lo: sum.1,
        } + Wide::from(f64::from(chunk) * 2.0_f64.powi(exponent));
        (value.hi, value.lo)
    })
}

/// Fixed-point Airy phase with the fractional low part kept for a cancellation-resistant
/// oscillatory value near a zero.
pub(crate) fn airy_oscillatory_phase_parts(x: f64) -> Result<(f64, f64), SciOxideError> {
    crate::support::check_positive(
        x,
        scioxide_core::argument!("x"),
        "positive finite phase argument required",
    )?;
    let (significand, exponent) = crate::support::wide::finite_parts(x);
    let mantissa = (significand * 4_503_599_627_370_496.0) as u64;
    let exponent = i64::from(exponent) - 52;
    let odd_exponent = exponent.rem_euclid(2) != 0;
    let square_root_argument = if odd_exponent {
        mantissa << 1
    } else {
        mantissa
    };
    let power = exponent + exponent.div_euclid(2);
    let root = fixed_sqrt(square_root_argument);
    let mantissa_product = multiply_mantissa(&root, mantissa);
    let fixed_product = multiply_reciprocal(&mantissa_product);
    let binary_point =
        i64::try_from(SQRT_FRACTION_BITS + RECIPROCAL_FRACTION_BITS).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("airy_fixed_point_phase"))
        })? - power;
    let (cycles_hi, cycles_lo) = leading_fraction(&fixed_product, binary_point);
    let phase = Wide {
        hi: cycles_hi,
        lo: cycles_lo,
    } * crate::support::constants::TAU
        + crate::support::constants::FRAC_PI_4;
    Ok((phase.hi, phase.lo))
}

/// Exact-input reduction of 2 sqrt(x) modulo 2*pi. The existing fixed square
/// root and reciprocal 1/(3*pi) need only the integer multiplier three here.
pub(super) fn sqrt_oscillatory_phase(x: f64) -> Wide {
    let (significand, exponent) = crate::support::wide::finite_parts(x);
    let mantissa = (significand * 4_503_599_627_370_496.0) as u64;
    let exponent = i64::from(exponent) - 52;
    let root = fixed_sqrt(if exponent.rem_euclid(2) == 0 {
        mantissa
    } else {
        mantissa << 1
    });
    let product = multiply_reciprocal(&multiply_mantissa(&root, 3));
    let binary_point =
        (SQRT_FRACTION_BITS + RECIPROCAL_FRACTION_BITS) as i64 - exponent.div_euclid(2);
    let (hi, lo) = leading_fraction(&product, binary_point);
    Wide { hi, lo } * crate::support::constants::TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_phase_matches_direct_moderate_arguments() {
        [8.0_f64, 64.0, 1024.0].into_iter().for_each(|x| {
            let expected = ((2.0 / 3.0) * x * x.sqrt()).rem_euclid(core::f64::consts::TAU)
                + core::f64::consts::FRAC_PI_4;
            let (hi, lo) = airy_oscillatory_phase_parts(x).unwrap();
            let actual = hi + lo;
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "x={x}: {actual} vs {expected}"
            );
        });
    }

    #[test]
    fn fixed_phase_is_bounded_at_every_exponent_extreme() {
        [f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)]
            .into_iter()
            .for_each(|x| {
                let (hi, lo) = airy_oscillatory_phase_parts(x).unwrap();
                let phase = hi + lo;
                assert!(phase >= core::f64::consts::FRAC_PI_4);
                assert!(phase < core::f64::consts::TAU + core::f64::consts::FRAC_PI_4);
            });
    }

    #[test]
    fn fixed_phase_is_bounded_across_every_f64_normal_exponent_bin() {
        (1_u64..=0x7fe).for_each(|exponent| {
            let x = f64::from_bits(exponent << 52);
            let (hi, lo) = airy_oscillatory_phase_parts(x).unwrap();
            let phase = hi + lo;
            assert!(phase >= core::f64::consts::FRAC_PI_4, "exponent={exponent}");
            assert!(
                phase < core::f64::consts::TAU + core::f64::consts::FRAC_PI_4,
                "exponent={exponent}"
            );
        });
    }
}
