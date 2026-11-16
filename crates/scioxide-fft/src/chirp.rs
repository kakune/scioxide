//! Integral-index phase construction shared by chirp transforms.
use scioxide_core::{Argument, Complex, RealScalar, SciOxideError};

/// Multiply a binary floating-point number of turns by an integer modulo one.
/// Repeated doubling avoids rounding the integer (including squared indices).
pub(crate) fn turns_product(turns: f64, mut index: u128) -> f64 {
    // Double-and-add in Z/1: `term` holds (turns * 2^bit) mod 1 and `result`
    // accumulates the set bits of `index`. For dyadic `turns` (for example
    // 1/2^k) every sum and `% 1.0` is exact, so the result is exactly
    // (turns * index) mod 1 even when `index` (typically j^2) exceeds 2^53 and
    // could not itself be represented in f64.
    let mut term = turns % 1.0;
    let mut result = 0.0;
    while index != 0 {
        if index & 1 != 0 {
            result = wrap_turn(result + term);
        }
        term = wrap_turn(term + term);
        index >>= 1;
    }
    result
}

/// Exact remainder for the loop's values in (-2, 2), including negative zero.
#[inline]
fn wrap_turn(value: f64) -> f64 {
    if value >= 1.0 {
        value - 1.0
    } else if value <= -1.0 {
        -((-value) - 1.0)
    } else {
        value
    }
}

pub(crate) fn phase<S: RealScalar>(turns: f64, index: u128) -> Result<Complex<S>, SciOxideError> {
    let reduced = turns_product(turns, index);
    // Map the phase into [-1/2, 1/2] turns before scaling by 2*pi so the angle
    // given to sin_cos is at most pi in magnitude (the best-conditioned range),
    // and phases of conjugate indices stay symmetric.
    let centered = if reduced > 0.5 {
        reduced - 1.0
    } else if reduced < -0.5 {
        reduced + 1.0
    } else {
        reduced
    };
    let (sin, cos) = (core::f64::consts::TAU * centered).sin_cos();
    Ok(Complex::new(
        S::try_from_f64_lossy(cos)?,
        S::try_from_f64_lossy(sin)?,
    ))
}

/// Return the exact Bluestein exponent `index^2 mod 2n` without converting an
/// index or length through the destination floating-point type.
///
/// An out-of-range index is attributed to `argument`, the caller parameter that determined `n`.
pub(crate) fn square_mod_twice_len(
    index: usize,
    n: usize,
    argument: Argument,
) -> Result<u128, SciOxideError> {
    if n == 0 || index >= n {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "the chirp length must be non-zero and the index must be smaller than the length",
        ));
    }
    let modulus = (n as u128).checked_mul(2).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("bluestein_chirp_modulus"))
    })?;
    let square = (index as u128).checked_mul(index as u128).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("bluestein_chirp_square"))
    })?;
    // The Bluestein chirp is exp(-i*pi*j^2/n) = exp(-2*pi*i * (j^2 mod 2n)/(2n)),
    // so reducing modulo 2n (not n) preserves the half-turn parity of j^2.
    Ok(square % modulus)
}

#[cfg(test)]
mod tests {
    use super::turns_product;
    use scioxide_core::RealScalar;

    #[test]
    fn exact_wrap_matches_remainder_bits() {
        for value in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0 - f64::EPSILON,
            -1.0 + f64::EPSILON,
            2.0 - f64::EPSILON,
            -2.0 + f64::EPSILON,
        ] {
            assert_eq!(super::wrap_turn(value).to_bits(), (value % 1.0).to_bits());
        }
        let reference = |turns: f64, mut index: u128| {
            let mut term = turns % 1.0;
            let mut result = 0.0;
            while index != 0 {
                if index & 1 != 0 {
                    result = (result + term) % 1.0;
                }
                term = (term + term) % 1.0;
                index >>= 1;
            }
            result
        };
        let mut state = 0x173a_f129_844e_213d_u64;
        for _ in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let turns = f64::from_bits(state);
            if !turns.is_finite() {
                continue;
            }
            let index = u128::from(state) * u128::from(state.rotate_left(29));
            assert_eq!(
                turns_product(turns, index).to_bits(),
                reference(turns, index).to_bits()
            );
        }
        for turns in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0 - f64::EPSILON,
            -2.0 + f64::EPSILON,
        ] {
            for index in [0, 1, 2, 63, u128::MAX] {
                assert_eq!(
                    turns_product(turns, index).to_bits(),
                    reference(turns, index).to_bits()
                );
            }
        }
    }

    #[test]
    fn negative_turn_phases_are_centered_and_conjugate() {
        fn check<S: RealScalar>() {
            for turns in [0.0, 0.125, 0.5, 0.75, 1.0 - f64::EPSILON, 1.125] {
                for index in [1, 3, (1_u128 << 53) + 17, u128::MAX] {
                    let positive = super::phase::<S>(turns, index).unwrap();
                    let negative = super::phase::<S>(-turns, index).unwrap();
                    // Centred arguments must agree even near a complete turn,
                    // where evaluating phases near -2*pi must preserve symmetry.
                    assert_eq!(positive.re, negative.re);
                    assert_eq!(positive.im, -negative.im);
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn phases_preserve_squared_indices_beyond_scalar_integer_precision() {
        for j in [(1_u128 << 24) + 1, (1_u128 << 53) + 17, usize::MAX as u128] {
            let square = j * j;
            let expected = (square % 256) as f64 / 256.0;
            assert_eq!(turns_product(1.0 / 256.0, square), expected);
            assert_eq!(turns_product(-1.0 / 256.0, square), -expected);
        }
    }
}
