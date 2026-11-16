//! Filtered, resource-bounded exact dyadic determinants in arbitrary dimension.
//!
//! Like `exact`, this accumulates signed products of decoded IEEE significands.
//! The permutation expansion is bounded by the caller's work limit.
//! Exact fallback expands lifted norms into monomials without rounding them.
use super::exact::{add_product, compare_bins, multiply_words};
use alloc::vec::Vec;
use scioxide_core::SciOxideError;
use scioxide_core::backend::{checked_add, checked_mul};

/// Scratch is sized from the maximum determinant order; no per-term allocation.
#[derive(Debug)]
pub(crate) struct ExactWorkspace {
    permutation: Vec<usize>,
    product: Vec<u64>,
    positive: Vec<u64>,
    negative: Vec<u64>,
    determinants: Vec<f64>,
    permanents: Vec<f64>,
    lifts: Vec<f64>,
    order: usize,
}
impl ExactWorkspace {
    // Each IEEE factor spans binary exponents [-1074, 1024), about 2100 bits,
    // so a product of `order` factors needs 2100 * order bits. The four
    // extra limbs (256 bits) absorb the carry from any usize-countable number
    // of terms. `product` needs one limb per factor, since each significand
    // has fewer than 64 bits.
    //
    // Size arithmetic
    pub(crate) fn bytes(order: usize) -> Result<usize, SciOxideError> {
        let bits = checked_mul(
            2100,
            order,
            scioxide_core::quantity!("exact_determinant_bits"),
        )?;
        let limbs = checked_add(
            bits / 64,
            4,
            scioxide_core::quantity!("exact_determinant_limbs"),
        )?;
        checked_mul(
            checked_add(
                checked_mul(limbs, 2, scioxide_core::quantity!("exact_bins"))?,
                checked_add(
                    checked_mul(order, 3, scioxide_core::quantity!("exact_product"))?,
                    2 * subsets(order),
                    scioxide_core::quantity!("predicate_filter_storage"),
                )?,
                scioxide_core::quantity!("exact_storage"),
            )?,
            8,
            scioxide_core::quantity!("exact_bytes"),
        )
    }
    pub(crate) fn new(order: usize) -> Result<Self, SciOxideError> {
        let mut result = Self::empty();
        result.try_reserve(order)?;
        Ok(result)
    }
    pub(crate) const fn empty() -> Self {
        Self {
            permutation: Vec::new(),
            product: Vec::new(),
            positive: Vec::new(),
            negative: Vec::new(),
            determinants: Vec::new(),
            permanents: Vec::new(),
            lifts: Vec::new(),
            order: 0,
        }
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.permutation.capacity() * size_of::<usize>()
            + (self.product.capacity()
                + self.positive.capacity()
                + self.negative.capacity()
                + self.determinants.capacity()
                + self.permanents.capacity()
                + self.lifts.capacity())
                * size_of::<u64>()
    }
    pub(crate) fn projected_bytes(&self, order: usize) -> Result<usize, SciOxideError> {
        Self::bytes(order)?;
        let limbs = 2100 * order / 64 + 4;
        Ok(self.permutation.capacity().max(order) * size_of::<usize>()
            + (self.product.capacity().max(order)
                + self.positive.capacity().max(limbs)
                + self.negative.capacity().max(limbs)
                + self.determinants.capacity().max(subsets(order))
                + self.permanents.capacity().max(subsets(order))
                + self.lifts.capacity().max(order))
                * size_of::<u64>())
    }
    pub(crate) fn try_reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        Self::bytes(order)?;
        if order <= self.order {
            return Ok(());
        }
        let limbs = 2100 * order / 64 + 4;
        crate::workspace::grow(&mut self.permutation, order, 0)?;
        crate::workspace::grow(&mut self.product, order, 0)?;
        crate::workspace::grow(&mut self.positive, limbs, 0)?;
        crate::workspace::grow(&mut self.negative, limbs, 0)?;
        crate::workspace::grow(&mut self.determinants, subsets(order), 0.)?;
        crate::workspace::grow(&mut self.permanents, subsets(order), 0.)?;
        crate::workspace::grow(&mut self.lifts, order, 0.)?;
        self.order = order;
        Ok(())
    }
    /// Rows contain coordinates only; the last column is implicit one.
    /// Counts one sign evaluation, then the exact monomial bound only on
    /// fallback (including any zero products pruned without arithmetic).
    pub(crate) fn sign(
        &mut self,
        rows: &[f64],
        dimension: usize,
        lifted: bool,
        admit: &mut impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<i8, SciOxideError> {
        self.filtered(rows, dimension, lifted, false, admit)
    }
    pub(crate) fn matrix_sign(
        &mut self,
        rows: &[f64],
        order: usize,
        admit: &mut impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<i8, SciOxideError> {
        self.filtered(rows, order, false, true, admit)
    }
    fn filtered(
        &mut self,
        rows: &[f64],
        dimension: usize,
        lifted: bool,
        matrix: bool,
        admit: &mut impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<i8, SciOxideError> {
        let determinant = Determinant::checked(rows, dimension, lifted, matrix, self.order)?;
        admit(1)?;
        if let Some(sign) = self.filter(determinant) {
            return Ok(sign);
        }
        self.evaluate(determinant, admit)
    }
    fn filter(&mut self, determinant: Determinant<'_>) -> Option<i8> {
        let Determinant {
            rows,
            dimension,
            lifted,
            matrix,
            order,
        } = determinant;
        if order > FILTER_ORDER || rows.iter().any(|&x| !in_range(x)) {
            return None;
        }
        if lifted {
            for (lift, row) in self.lifts.iter_mut().zip(rows.chunks_exact(dimension)) {
                *lift = row.iter().fold(0., |sum, x| sum + x * x);
            }
        }
        self.determinants[0] = 1.;
        self.permanents[0] = 1.;
        // Expand the last row of each prefix minor. Each column subset is
        // evaluated once: O(n * 2^n) arithmetic, rather than n! products.
        for mask in 1usize..1 << order {
            let row = mask.count_ones() as usize - 1;
            let mut columns = mask;
            let mut negative = !row.is_multiple_of(2);
            let mut det = 0.;
            let mut permanent = 0.;
            while columns != 0 {
                let column = columns.trailing_zeros() as usize;
                let bit = 1 << column;
                columns ^= bit;
                let value = if matrix || column < dimension {
                    rows[row * dimension + column]
                } else if lifted && column == dimension {
                    self.lifts[row]
                } else {
                    1.
                };
                let product = value * self.determinants[mask ^ bit];
                det += if negative { -product } else { product };
                permanent += value.abs() * self.permanents[mask ^ bit];
                negative = !negative;
            }
            self.determinants[mask] = det;
            self.permanents[mask] = permanent;
        }
        let last = (1 << order) - 1;
        let det = self.determinants[last];
        let error = 256. * f64::EPSILON * self.permanents[last];
        if det > error {
            Some(1)
        } else if det < -error {
            Some(-1)
        } else {
            None
        }
    }
    fn evaluate(
        &mut self,
        determinant: Determinant<'_>,
        admit: &mut impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<i8, SciOxideError> {
        self.positive.fill(0);
        self.negative.fill(0);
        self.expand(determinant, 0, 0, false, admit)?;
        Ok(match compare_bins(&self.positive, &self.negative) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        })
    }
    fn expand(
        &mut self,
        determinant: Determinant<'_>,
        row: usize,
        lift_axis: usize,
        negative: bool,
        admit: &mut impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let Determinant {
            rows,
            dimension,
            lifted,
            matrix,
            order,
        } = determinant;
        if row == order {
            admit(1)?;
            self.product.fill(0);
            self.product[0] = 1;
            let mut exponent = 0;
            let mut negative = negative;
            for (row, &column) in self.permutation[..order].iter().enumerate() {
                if matrix || column < dimension {
                    multiply(
                        &mut self.product,
                        rows[row * dimension + column],
                        &mut exponent,
                        &mut negative,
                    );
                } else if lifted && column == dimension {
                    for _ in 0..2 {
                        multiply(
                            &mut self.product,
                            rows[row * dimension + lift_axis],
                            &mut exponent,
                            &mut negative,
                        );
                    }
                }
            }
            // Products contain at most `order` factors. Bins also cover the
            // carry from any usize-representable number of admitted terms.
            let offset = usize::try_from(exponent + 1074 * self.order as i64).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("exact_exponent"))
            })?;
            let bins = if negative {
                &mut self.negative
            } else {
                &mut self.positive
            };
            add_product(bins, &self.product, offset);
            return Ok(());
        }
        for column in 0..order {
            if self.permutation[..row].contains(&column) {
                continue;
            }
            let negative = negative
                ^ !self.permutation[..row]
                    .iter()
                    .filter(|&&previous| previous > column)
                    .count()
                    .is_multiple_of(2);
            self.permutation[row] = column;
            if !matrix && lifted && column == dimension {
                for axis in 0..dimension {
                    if rows[row * dimension + axis] == 0. {
                        admit(subtree_terms(order - row - 1, 1)?)?;
                    } else {
                        self.expand(determinant, row + 1, axis, negative, admit)?;
                    }
                }
            } else if (matrix || column < dimension) && rows[row * dimension + column] == 0. {
                // Prune a zero product without doing its limb arithmetic, but
                // admit its entire monomial bound, preserving budget boundaries.
                let axes = if lifted && !self.permutation[..=row].contains(&dimension) {
                    dimension
                } else {
                    1
                };
                admit(subtree_terms(order - row - 1, axes)?)?;
            } else {
                self.expand(determinant, row + 1, lift_axis, negative, admit)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Determinant<'a> {
    rows: &'a [f64],
    dimension: usize,
    lifted: bool,
    matrix: bool,
    order: usize,
}
impl<'a> Determinant<'a> {
    fn checked(
        rows: &'a [f64],
        dimension: usize,
        lifted: bool,
        matrix: bool,
        maximum: usize,
    ) -> Result<Self, SciOxideError> {
        let order = if matrix {
            dimension
        } else {
            checked_add(
                dimension,
                1 + usize::from(lifted),
                scioxide_core::quantity!("exact_determinant_shape"),
            )?
        };
        if order == 0
            || order > maximum
            || rows.len()
                != checked_mul(
                    order,
                    dimension,
                    scioxide_core::quantity!("exact_determinant_shape"),
                )?
        {
            return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "exact_determinant_shape"
            )));
        }
        Ok(Self {
            rows,
            dimension,
            lifted,
            matrix,
            order,
        })
    }
}
fn subtree_terms(remaining: usize, axes: usize) -> Result<usize, SciOxideError> {
    (1..=remaining).try_fold(axes, |terms, factor| {
        checked_mul(
            terms,
            factor,
            scioxide_core::quantity!("exact_determinant_terms"),
        )
    })
}

fn multiply(words: &mut [u64], value: f64, exponent: &mut i64, negative: &mut bool) {
    let (power, sign) = multiply_words(words, value);
    *exponent += i64::from(power);
    *negative ^= sign;
}

// The standard relative-error model applies throughout this envelope:
// inputs are multiples of 2^-92, all polynomials have degree <= 10, so
// every nonzero intermediate is a multiple of 2^-920 (normal). Absolute
// monomials are <= 8 * 2^400 and their sums <= 10! times that bound.
// The permanent and its error bound therefore cannot overflow/underflow.
//
// Write u = EPSILON/2 and gamma_k = k*u/(1-k*u). At prefix order k,
// one multiplication and at most k additions add k+1 rounding factors
// per expanded monomial. Across n <= 10 levels their depth is at most
// n*(n+3)/2 <= 65. A lifted entry adds at most d <= 8 factors for its
// sum of squares. Both the determinant error and the computed permanent
// error are bounded by gamma_73 times the exact permanent P. Hence
// |det_hat-det| <= gamma_73/(1-gamma_73) * P_hat < 512*u*P_hat.
// 256*EPSILON is an exact power of two; scaling the normal permanent by
// it is exact. Strict separation certifies a sign, never exact zero.
const FILTER_ORDER: usize = 10;
const fn subsets(order: usize) -> usize {
    1 << if order < FILTER_ORDER {
        order
    } else {
        FILTER_ORDER
    }
}
fn in_range(value: f64) -> bool {
    super::filter::in_envelope(value, 40)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use num::{BigRational, One, Zero};

    // Independent rational Gaussian elimination: no production decoder,
    // recurrence, error bound or exact accumulator participates in the oracle.
    fn reference(rows: &[f64], dimension: usize, lifted: bool) -> i8 {
        let matrix: Vec<Vec<BigRational>> = rows
            .chunks_exact(dimension)
            .map(|point| {
                let mut row: Vec<_> = point
                    .iter()
                    .map(|&x| BigRational::from_float(x).unwrap())
                    .collect();
                if lifted {
                    row.push(row.iter().map(|x| x * x).sum());
                }
                row.push(BigRational::one());
                row
            })
            .collect();
        rational_determinant(matrix)
    }
    #[allow(clippy::needless_range_loop)]
    fn rational_determinant(mut matrix: Vec<Vec<BigRational>>) -> i8 {
        let mut det = BigRational::one();
        for column in 0..matrix.len() {
            let Some(pivot) = (column..matrix.len()).find(|&i| !matrix[i][column].is_zero()) else {
                return 0;
            };
            if pivot != column {
                matrix.swap(pivot, column);
                det = -det;
            }
            let diagonal = matrix[column][column].clone();
            det *= &diagonal;
            for row in column + 1..matrix.len() {
                let factor = &matrix[row][column] / &diagonal;
                for j in column + 1..matrix.len() {
                    let reduction = &factor * &matrix[column][j];
                    matrix[row][j] -= reduction;
                }
                matrix[row][column] = BigRational::zero();
            }
        }
        match det.cmp(&BigRational::zero()) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        }
    }
    fn check(rows: &[f64], dimension: usize, lifted: bool) -> bool {
        let order = dimension + 1 + usize::from(lifted);
        let mut workspace = ExactWorkspace::new(order).unwrap();
        let expected = reference(rows, dimension, lifted);
        let determinant = Determinant::checked(rows, dimension, lifted, false, order).unwrap();
        let certified = workspace.filter(determinant);
        if let Some(sign) = certified {
            assert_eq!(sign, expected);
        }
        let mut terms = 0;
        assert_eq!(
            workspace
                .sign(rows, dimension, lifted, &mut |count| {
                    terms += count;
                    Ok(())
                })
                .unwrap(),
            expected,
            "d={dimension}, lifted={lifted}, rows={rows:?}"
        );
        let bound = subtree_terms(order, if lifted { dimension } else { 1 }).unwrap();
        assert_eq!(terms, if certified.is_some() { 1 } else { 1 + bound });
        certified.is_some()
    }
    fn random(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / 9007199254740992.
    }
    #[test]
    fn nd_random_and_envelope_endpoints_match_rational_oracles() {
        let mut state = 1923;
        for dimension in 4..=8 {
            for lifted in [false, true] {
                let order = dimension + 1 + usize::from(lifted);
                for case in 0..16 {
                    let mut rows: Vec<_> = (0..order * dimension)
                        .map(|_| {
                            let x = random(&mut state);
                            match case % 4 {
                                0 => (1. + x) * 2_f64.powi(-40),
                                1 => -(1. + x) * 2_f64.powi(39),
                                _ => 2. * x - 1.,
                            }
                        })
                        .collect();
                    assert!(check(&rows, dimension, lifted));
                    let narrow: Vec<_> = rows.iter().map(|&x| f64::from(x as f32)).collect();
                    assert!(check(&narrow, dimension, lifted));
                    // Odd and even row permutations retain the oracle convention.
                    for axis in 0..dimension {
                        rows.swap(axis, dimension + axis);
                    }
                    assert!(check(&rows, dimension, lifted));
                    rows.rotate_left(2 * dimension);
                    assert!(check(&rows, dimension, lifted));
                }
            }
        }
    }
    #[test]
    fn nd_adjacent_float_degeneracies_match_rational_oracles() {
        for dimension in 4..=8 {
            for radius in [
                1_f64.next_down(),
                1.,
                1_f64.next_up(),
                f64::from(1_f32.next_down()),
                f64::from(1_f32.next_up()),
            ] {
                // A thin simplex about the affine plane x[d-1] = 1.
                let mut rows = alloc::vec![0.; (dimension + 1)*dimension];
                for row in 0..dimension {
                    rows[row * dimension + dimension - 1] = 1.;
                    if row > 0 {
                        rows[row * dimension + row - 1] = 1.;
                    }
                }
                rows[dimension * dimension] = 0.25;
                rows[dimension * dimension + dimension - 1] = radius;
                let certified = check(&rows, dimension, false);
                if (radius - 1.).abs() <= f64::EPSILON {
                    assert!(!certified);
                }
                // A full-dimensional sphere through e_i and -e_1, with the
                // last site's radius perturbed by one ulp in either width.
                let mut rows = alloc::vec![0.; (dimension + 2)*dimension];
                for axis in 0..dimension {
                    rows[axis * dimension + axis] = 1.;
                }
                rows[dimension * dimension] = -1.;
                rows[(dimension + 1) * dimension + 1] = -radius;
                let certified = check(&rows, dimension, true);
                if (radius - 1.).abs() <= f64::EPSILON {
                    assert!(!certified);
                }
            }
        }
    }
    #[test]
    fn nd_extreme_exponents_and_duplicates_match_rational_oracles() {
        let scales = [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            2_f64.powi(-40).next_down(),
            2_f64.powi(40).next_up(),
            f64::MAX,
            f64::from(f32::from_bits(1)),
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
        ];
        for dimension in 4..=8 {
            for (case, &scale) in scales.iter().enumerate() {
                for lifted in [false, true] {
                    let order = dimension + 1 + usize::from(lifted);
                    let mut rows = alloc::vec![0.; order*dimension];
                    for axis in 0..dimension {
                        rows[axis * dimension + axis] = if axis.is_multiple_of(2) {
                            scale
                        } else {
                            scales[(case + 1) % scales.len()]
                        };
                    }
                    rows[dimension * dimension] = -scale;
                    if lifted {
                        rows[(dimension + 1) * dimension + 1] = -1.;
                    }
                    assert!(!check(&rows, dimension, lifted));
                    rows[..dimension].fill(0.);
                    rows[dimension..2 * dimension].fill(0.);
                    assert!(!check(&rows, dimension, lifted));
                }
            }
        }
    }
    #[test]
    fn unsupported_orders_and_invalid_shapes_stay_bounded() {
        let dimension = 10;
        let mut rows = alloc::vec![0.; dimension * (dimension + 1)];
        for axis in 0..dimension {
            rows[(axis + 1) * dimension + axis] = 1.;
        }
        assert!(!check(&rows, dimension, false));
        let mut workspace = ExactWorkspace::new(2).unwrap();
        assert!(workspace.sign(&[], 1, false, &mut |_| Ok(())).is_err());
        assert!(workspace.matrix_sign(&[], 0, &mut |_| Ok(())).is_err());
        assert!(workspace.sign(&[0.; 6], 2, false, &mut |_| Ok(())).is_err());
    }

    #[test]
    fn homogeneous_filters_through_order_ten_match_rationals() {
        let mut state = 0x1923_1234;
        for order in 1..=10 {
            let mut workspace = ExactWorkspace::new(order).unwrap();
            for _ in 0..8 {
                let rows: Vec<_> = (0..order * order)
                    .map(|_| 2. * random(&mut state) - 1.)
                    .collect();
                let expected = rational_determinant(
                    rows.chunks_exact(order)
                        .map(|row| {
                            row.iter()
                                .map(|&x| BigRational::from_float(x).unwrap())
                                .collect()
                        })
                        .collect(),
                );
                let mut terms = 0;
                assert_eq!(
                    workspace
                        .matrix_sign(&rows, order, &mut |count| {
                            terms += count;
                            Ok(())
                        })
                        .unwrap(),
                    expected
                );
                assert_eq!(terms, 1);
            }
        }
    }

    #[test]
    fn homogeneous_filter_matches_exact_dyadic_determinants_and_charges() {
        for order in 2..=4 {
            let mut workspace = ExactWorkspace::new(order).unwrap();
            let mut state = 42_u64;
            for case in 0..300 {
                let mut rows = [0.; 16];
                for value in &mut rows[..order * order] {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *value = ((state >> 32) as i32 as f64) / 2147483648.;
                }
                if case % 3 == 0 {
                    for axis in 0..order {
                        rows[order + axis] = rows[axis];
                    }
                    if case % 2 == 0 {
                        rows[order] += f64::EPSILON;
                    }
                }
                if case % 7 == 0 {
                    rows[0] = f64::MIN_POSITIVE;
                }
                if case % 11 == 0 {
                    rows[1] = f64::MAX;
                }
                if case % 5 == 0 {
                    for row in rows[..order * order].chunks_exact_mut(order) {
                        row[order - 1] = if case % 2 == 0 { -2. } else { 1. };
                    }
                }
                let rows = &rows[..order * order];
                let mut exact_terms = 0;
                let expected = workspace
                    .evaluate(
                        Determinant::checked(rows, order, false, true, order).unwrap(),
                        &mut |count| {
                            exact_terms += count;
                            Ok(())
                        },
                    )
                    .unwrap();
                let mut filtered_terms = 0;
                let actual = workspace
                    .matrix_sign(rows, order, &mut |count| {
                        filtered_terms += count;
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(actual, expected);
                assert!(filtered_terms == 1 || filtered_terms == exact_terms + 1);
            }
        }
    }
}
