//! Exact dyadic determinant, with no floating arithmetic on coordinate values.
//!
//! At most five 53-bit significands form a term (265 bits). Its exponent is
//! at least -5370. Each term has magnitude < 2^5120, and there are at most
//! 5! * 3 = 360 terms. Thus either unsigned sum is < 2^5129; 168 u64 limbs
//! anchored at -5370 cover 10752 bits, exceeding the required 10499 bits.

use super::PredicateSign;
use scioxide_core::RealScalar;

const LIMBS: usize = 168;
const MIN_EXPONENT: i32 = -5370;

// One Leibniz monomial held as an exact integer times 2^exponent. Five limbs
// hold the product of at most five 53-bit significands without truncation.
#[derive(Clone, Copy)]
struct Product {
    words: [u64; 5],
    exponent: i32,
    negative: bool,
}

impl Product {
    const ONE: Self = Self {
        words: [1, 0, 0, 0, 0],
        exponent: 0,
        negative: false,
    };

    fn times(mut self, value: f64) -> Self {
        let (exponent, negative) = multiply_words(&mut self.words, value);
        self.exponent += exponent;
        self.negative ^= negative;
        self
    }
}

// Positive and negative monomials go into separate unsigned fixed-point
// bins, so accumulation is only carry-propagating addition, with no signed
// borrows. Bit i of each bin has weight 2^(MIN_EXPONENT + i). The sign
// comes from comparing the two sums, most significant limb first.
struct Accumulator {
    positive: [u64; LIMBS],
    negative: [u64; LIMBS],
}

// The carry chain stops at the first limb that does not overflow. The
// module-level bound guarantees this happens before the end of the bin.
pub(super) fn add_word(words: &mut [u64], mut index: usize, mut carry: u64) {
    while carry != 0 {
        let (value, overflow) = words[index].overflowing_add(carry);
        words[index] = value;
        carry = u64::from(overflow);
        index += 1;
    }
}

impl Accumulator {
    fn add(&mut self, product: Product) {
        let words = if product.negative {
            &mut self.negative
        } else {
            &mut self.positive
        };
        // A product's exponent is never below MIN_EXPONENT, so `offset` is
        // non-negative. Each limb is split across two bin limbs when the
        // offset is not limb-aligned.
        let offset = (product.exponent - MIN_EXPONENT) as usize;
        add_product(words, &product.words, offset);
    }

    // Leibniz determinant expansion. Column inversions determine parity; the
    // lifted column branches into exact coordinate-square monomials.
    //
    // The matrix is untranslated, with rows [coordinates, (norm²), 1] and
    // `points.len()` columns. A column past the coordinates (and past the
    // lifted column) is the column of ones, which contributes factor 1.
    // `used` is a bitset of columns taken by earlier rows. The used columns
    // greater than `column` are exactly the inversions that choosing
    // `column` now adds, so parity builds up incrementally. A zero
    // coordinate prunes its whole subtree, because every monomial under it
    // is zero.
    fn expand(
        &mut self,
        points: &[[f64; 3]],
        dimension: usize,
        row: usize,
        used: u8,
        product: Product,
        lifted: bool,
    ) {
        if row == points.len() {
            self.add(product);
            return;
        }
        for column in 0..points.len() {
            if used & (1 << column) != 0 {
                continue;
            }
            let mut term = product;
            term.negative ^= !(used >> (column + 1)).count_ones().is_multiple_of(2);
            let next_used = used | (1 << column);
            if column < dimension {
                let coordinate = points[row][column];
                if coordinate != 0.0 {
                    self.expand(
                        points,
                        dimension,
                        row + 1,
                        next_used,
                        term.times(coordinate),
                        lifted,
                    );
                }
            } else if lifted && column == dimension {
                for &coordinate in &points[row][..dimension] {
                    if coordinate != 0.0 {
                        self.expand(
                            points,
                            dimension,
                            row + 1,
                            next_used,
                            term.times(coordinate).times(coordinate),
                            lifted,
                        );
                    }
                }
            } else {
                self.expand(points, dimension, row + 1, next_used, term, lifted);
            }
        }
    }
}

// Envelope misses may include nonfinite input: validate before exact arithmetic.
#[cold]
#[inline(never)]
pub(super) fn checked_fixed<S: scioxide_core::RealScalar, const D: usize, const N: usize>(
    points: [&[S; D]; N],
    lifted: bool,
) -> Result<PredicateSign, scioxide_core::SciOxideError> {
    let mut coordinates = [[0.0; 3]; 5];
    for (destination, point) in coordinates.iter_mut().zip(points) {
        for (output, &value) in destination.iter_mut().zip(point) {
            if !value.is_finite() {
                return Err(scioxide_core::SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "coordinates must be finite",
                ));
            }
            *output = value.into_f64();
        }
    }
    Ok(sign(&coordinates, D, N, lifted))
}

// Padding belongs exclusively to the out-of-line exact fallback.
#[cold]
#[inline(never)]
pub(super) fn fixed<S: RealScalar, const D: usize, const N: usize>(
    points: [&[S; D]; N],
    lifted: bool,
) -> PredicateSign {
    let mut coordinates = [[0.0; 3]; 5];
    for (destination, point) in coordinates.iter_mut().zip(points) {
        for (value, &x) in destination.iter_mut().zip(point) {
            *value = x.into_f64();
        }
    }
    sign(&coordinates, D, N, lifted)
}

pub(super) fn sign(
    points: &[[f64; 3]; 5],
    dimension: usize,
    count: usize,
    lifted: bool,
) -> PredicateSign {
    let mut accumulator = Accumulator {
        positive: [0; LIMBS],
        negative: [0; LIMBS],
    };
    accumulator.expand(&points[..count], dimension, 0, 0, Product::ONE, lifted);
    compare_bins(&accumulator.positive, &accumulator.negative).into()
}

// Subtract exact bins before rounding. This shares determinant arithmetic with
// signs without allowing a rounded magnitude to decide topology.
pub(crate) fn magnitude(points: &[[f64; 3]; 5], dimension: usize, count: usize) -> (f64, i32) {
    let mut accumulator = Accumulator {
        positive: [0; LIMBS],
        negative: [0; LIMBS],
    };
    accumulator.expand(&points[..count], dimension, 0, 0, Product::ONE, false);
    // Subtract the smaller bin from the larger one so the unsigned
    // difference is |det| exactly. Callers need only the magnitude.
    let (greater, lesser) = if compare_bins(&accumulator.positive, &accumulator.negative).is_lt() {
        (&accumulator.negative, &accumulator.positive)
    } else {
        (&accumulator.positive, &accumulator.negative)
    };
    let mut magnitude = [0_u64; LIMBS];
    let mut borrow = false;
    for (out, (&a, &b)) in magnitude.iter_mut().zip(greater.iter().zip(lesser)) {
        let (difference, first) = a.overflowing_sub(b);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        *out = difference;
        borrow = first || second;
    }
    let Some(word) = magnitude.iter().rposition(|&word| word != 0) else {
        return (0.0, 0);
    };
    // Take the leading 53 bits, then round to nearest with
    // ties to even using guard and sticky bits. The result is a significand
    // in [1, 2] plus a separate binary exponent, because |det| can lie far
    // outside the f64 range (up to about 2^5129). A round-up carry that gives
    // exactly 2 is still exact as a value.
    let top = word * 64 + 63 - magnitude[word].leading_zeros() as usize;
    let bit = |position: usize| (magnitude[position / 64] >> (position % 64)) & 1;
    let bottom = top.saturating_sub(52);
    let mut significand = 0_u64;
    for position in (bottom..=top).rev() {
        significand = (significand << 1) | bit(position);
    }
    if bottom > 0 {
        let guard = bit(bottom - 1) != 0;
        let sticky = (0..bottom - 1).any(|position| bit(position) != 0);
        if guard && (sticky || significand & 1 != 0) {
            significand += 1;
        }
    }
    (
        (significand as f64).scalbn(-((top - bottom) as i32)),
        MIN_EXPONENT + top as i32,
    )
}

// Shared exact significand multiplication for fixed and dynamic predicates.
// Each caller sizes `words` so that the top limb cannot overflow. The final
// carry is therefore zero, and it is only debug-asserted.
pub(super) fn multiply_words(words: &mut [u64], value: f64) -> (i32, bool) {
    let scioxide_core::FloatDecomposition::Finite {
        significand,
        exponent,
        negative,
        ..
    } = value.decompose()
    else {
        unreachable!("caller established a finite scalar before decomposition");
    };
    let sign = if negative { -1_i8 } else { 1_i8 };
    let mut carry = 0_u128;
    for word in words {
        let product = u128::from(*word) * u128::from(significand) + carry;
        *word = product as u64;
        carry = product >> 64;
    }
    debug_assert_eq!(carry, 0);
    (i32::from(exponent), sign < 0)
}

// Callers prove that bins cover every shifted limb and its carry.
pub(super) fn add_product(bins: &mut [u64], product: &[u64], offset: usize) {
    let shift = offset % 64;
    for (i, &word) in product.iter().enumerate() {
        let index = offset / 64 + i;
        add_word(bins, index, word << shift);
        if shift != 0 {
            add_word(bins, index + 1, word >> (64 - shift));
        }
    }
}
pub(super) fn compare_bins(positive: &[u64], negative: &[u64]) -> core::cmp::Ordering {
    positive.iter().rev().cmp(negative.iter().rev())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_and_full_product_width() {
        let mut words = [u64::MAX; LIMBS];
        words[4] = 0;
        add_word(&mut words, 0, 1);
        assert_eq!(&words[..5], &[0, 0, 0, 0, 1]);
        let mut product = Product::ONE;
        for _ in 0..5 {
            product = product.times(f64::MAX);
        }
        assert_ne!(product.words[4], 0);
        let mut accumulator = Accumulator {
            positive: [0; LIMBS],
            negative: [0; LIMBS],
        };
        for _ in 0..360 {
            accumulator.add(product);
        }
        assert!(accumulator.positive.iter().any(|&x| x != 0));
        product.negative = true;
        for _ in 0..360 {
            accumulator.add(product);
        }
        assert_eq!(accumulator.positive, accumulator.negative);
        let tiny = Product::ONE
            .times(f64::from_bits(1))
            .times(f64::from_bits(1))
            .times(f64::from_bits(1))
            .times(f64::from_bits(1))
            .times(f64::from_bits(1));
        accumulator.add(tiny);
        assert_eq!(accumulator.positive[0], 1);
    }
}
