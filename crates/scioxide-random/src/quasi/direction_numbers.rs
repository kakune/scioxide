//! Direction numbers for the Sobol sequence.
//!
//! `data/new-joe-kuo-6.21201` is Joe and Kuo's D(6) table (2010-09-16) from
//! <https://web.maths.unsw.edu.au/~fkuo/sobol/>, embedded unchanged under the BSD
//! license in `data/LICENSE`. It defines dimensions 2–21,201; dimension one is the
//! van der Corput sequence. Only the requested dimensions are parsed and expanded
//! to 64 columns.

use alloc::vec::Vec;

use scioxide_core::SciOxideError;

/// Maximum number of bits used for Sobol sequence generation.
const MAX_BITS: u32 = 64;

/// Primitive polynomial and initial direction numbers for one Sobol coordinate.
///
/// A coordinate uses a primitive polynomial over GF(2) of degree `s`,
///
/// ```text
/// p(x) = xˢ + a₁ xˢ⁻¹ + … + aₛ₋₁ x + 1,
/// ```
///
/// encoded as the integer `a = a₁ a₂ … aₛ₋₁` in binary, without the leading and constant
/// coefficients (Bratley and Fox 1988). For example, degree 2 with `a = 1` is `x² + x + 1`. The
/// initial direction numbers `m₁, …, mₛ` must be odd with `mᵢ < 2ⁱ`. Later ones follow the
/// recurrence
///
/// ```text
/// mₖ = 2a₁ mₖ₋₁ ⊕ 2²a₂ mₖ₋₂ ⊕ … ⊕ 2ˢ⁻¹aₛ₋₁ mₖ₋ₛ₊₁ ⊕ 2ˢ mₖ₋ₛ ⊕ mₖ₋ₛ,
/// ```
///
/// where `⊕` is bitwise XOR, and the direction numbers are `vₖ = mₖ / 2ᵏ`.
///
/// # References
///
/// - P. Bratley and B. L. Fox, "Algorithm 659: implementing Sobol's quasirandom sequence
///   generator", *ACM Trans. Math. Softw.* 14(1), 1988, 88–100.
/// - S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
///   projections", *SIAM J. Sci. Comput.* 30(5), 2008, 2635–2654.
///
/// # Examples
///
/// ```
/// use scioxide_random::{DirectionNumberSet, QmcPlan, QuasiRng, SobolDirection};
/// // x^2 + x + 1 with m_1 = 1, m_2 = 3 defines the second coordinate.
/// let direction = SobolDirection::new(2, 1, vec![1, 3])?;
/// let source = DirectionNumberSet::Custom(vec![direction]);
/// let plan = QmcPlan::sobol_with_direction_numbers(2, source)?;
/// assert_eq!(plan.direction_words().map(<[_]>::len), Some(2));
/// let mut sequence = plan.sequence::<f64>()?;
/// sequence.skip(1)?;
/// assert_eq!(sequence.next_point()?, [0.5, 0.5]); // m_1 = 1 in every coordinate
/// assert!(SobolDirection::new(2, 1, vec![1, 4]).is_err()); // m_2 must be odd
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct SobolDirection {
    degree: u32,
    primitive_polynomial: u32,
    initial_direction_numbers: Vec<u32>,
}

impl SobolDirection {
    /// Creates a direction definition from the degree `s`, the polynomial code `a` and the
    /// initial direction numbers `m₁, …, mₛ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the degree is outside `1..=32`, if `a` does not encode a primitive
    ///   polynomial of that degree, or if there are not exactly `s` initial numbers, each odd
    ///   with `mᵢ < 2ⁱ`.
    pub fn new(
        degree: u32,
        primitive_polynomial: u32,
        initial_direction_numbers: Vec<u32>,
    ) -> Result<Self, SciOxideError> {
        validate_definition(degree, primitive_polynomial, &initial_direction_numbers)?;
        Ok(Self {
            degree,
            primitive_polynomial,
            initial_direction_numbers,
        })
    }

    /// Returns the degree `s` of the primitive polynomial.
    pub const fn degree(&self) -> u32 {
        self.degree
    }

    /// Returns the polynomial code `a`, the coefficients without the leading and constant ones.
    pub const fn primitive_polynomial(&self) -> u32 {
        self.primitive_polynomial
    }

    /// Returns the initial direction numbers `m₁, …, mₛ`.
    pub fn initial_direction_numbers(&self) -> &[u32] {
        &self.initial_direction_numbers
    }
}

/// Direction numbers for the Sobol sequence.
///
/// The first coordinate is always the van der Corput sequence in base 2. See
/// [`SobolDirection`] for an example.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum DirectionNumberSet {
    /// Joe and Kuo's D(6) table ("new-joe-kuo-6.21201", 2010), up to 21,201 dimensions.
    ///
    /// The numbers are chosen to improve the two-dimensional projections (Joe and Kuo 2008).
    JoeKuoD6,
    /// Your own definitions for coordinates 2, 3, …, one per coordinate.
    Custom(Vec<SobolDirection>),
}

/// Maximum dimensionality of the pinned Joe–Kuo D(6) source.
pub(super) const JOE_KUO_DIMENSIONS: usize = 21_201;

/// Pinned complete upstream table (BSD license in `data/LICENSE`).
const JOE_KUO_D6: &str = include_str!("data/new-joe-kuo-6.21201");

/// Generate the full direction matrix for `dim` Sobol coordinates.
pub(crate) fn generate_direction_numbers(
    dim: usize,
    set: &DirectionNumberSet,
) -> Result<Vec<[u64; MAX_BITS as usize]>, SciOxideError> {
    let available = match set {
        DirectionNumberSet::JoeKuoD6 => JOE_KUO_DIMENSIONS,
        DirectionNumberSet::Custom(definitions) => definitions.len().saturating_add(1),
    };
    if dim > available {
        return Err(super::plan::invalid(
            scioxide_core::argument!("dimensions"),
            "direction source has insufficient dimensions",
        ));
    }
    let mut result = scioxide_core::backend::try_with_capacity(
        dim,
        scioxide_core::quantity!("sobol_direction_numbers"),
    )?;
    if dim == 0 {
        return Ok(result);
    }

    // Dimension 0 is the van der Corput sequence (identity generator matrix),
    // which is why the direction tables start at the second dimension. All
    // direction numbers are stored left-aligned in 64 bits (v_k = m_k / 2^k),
    // so Gray-code updates are plain XORs of full-width words.
    result.push(core::array::from_fn(|index| {
        1_u64 << (MAX_BITS as usize - 1 - index)
    }));
    if dim == 1 {
        return Ok(result);
    }

    match set {
        DirectionNumberSet::JoeKuoD6 => {
            for line in JOE_KUO_D6.lines().skip(1).take(dim - 1) {
                let mut fields = line.split_whitespace();
                let mut read = || {
                    fields
                        .next()
                        .and_then(|s| s.parse::<u32>().ok())
                        .ok_or_else(|| {
                            super::plan::invalid(
                                scioxide_core::argument!("source"),
                                "malformed bundled direction table",
                            )
                        })
                };
                let _dimension = read()?;
                let degree = read()?;
                let polynomial = read()?;
                let mut initial = [0_u32; 32];
                for value in &mut initial[..degree as usize] {
                    *value = read()?;
                }
                result.push(build_direction_numbers(
                    degree,
                    polynomial,
                    &initial[..degree as usize],
                ));
            }
        }
        DirectionNumberSet::Custom(definitions) => {
            definitions
                .iter()
                .take(dim - 1)
                .map(|definition| {
                    build_direction_numbers(
                        definition.degree,
                        definition.primitive_polynomial,
                        &definition.initial_direction_numbers,
                    )
                })
                .for_each(|directions| result.push(directions));
        }
    }

    // A zero column would make the generator matrix singular and lose the
    // (t, s)-net property, so it is rejected as a malformed direction set.
    if result.iter().any(|directions| directions.contains(&0)) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("source"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "every generated Sobol direction column must be nonzero",
        ));
    }
    Ok(result)
}

fn validate_definition(degree: u32, polynomial: u32, initial: &[u32]) -> Result<(), SciOxideError> {
    if !(1..=32).contains(&degree) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("degree"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Sobol polynomial degree must be in 1..=32",
        ));
    }
    let coefficient_limit = 1_u64 << (degree - 1);
    if u64::from(polynomial) >= coefficient_limit || !is_primitive(degree, polynomial) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("source"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Sobol polynomial encoding must describe a primitive polynomial over GF(2)",
        ));
    }
    if initial.len() != degree as usize {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("source"),
            scioxide_core::InvalidInputReason::Inconsistent,
            alloc::format!(
                "degree {degree} requires {degree} initial direction numbers, received {}",
                initial.len()
            ),
        ));
    }
    if let Some((index, value)) = initial.iter().copied().enumerate().find(|(index, value)| {
        *value == 0 || *value & 1 == 0 || u64::from(*value) >= (1_u64 << (index + 1))
    }) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("source"),
            scioxide_core::InvalidInputReason::OutOfRange,
            alloc::format!(
                "m_{} must be nonzero, odd, and less than 2^{}; received {value}",
                index + 1,
                index + 1
            ),
        ));
    }
    Ok(())
}

fn is_primitive(degree: u32, encoded: u32) -> bool {
    // Joe-Kuo encoding stores only the interior coefficients; the leading and
    // constant terms are always 1. The polynomial is primitive iff x has
    // multiplicative order exactly 2^d - 1 in GF(2)[x]/(p): x^(2^d - 1) = 1 and
    // x^((2^d - 1)/q) != 1 for every prime factor q (trial division suffices
    // for d <= 32).
    let polynomial = (1_u64 << degree) | (u64::from(encoded) << 1) | 1;
    let order = (1_u64 << degree) - 1;
    if gf_pow(2, order, polynomial, degree) != 1 {
        return false;
    }
    let mut remaining = order;
    let mut candidate = 2;
    while candidate <= remaining / candidate {
        if remaining.is_multiple_of(candidate) {
            if gf_pow(2, order / candidate, polynomial, degree) == 1 {
                return false;
            }
            while remaining.is_multiple_of(candidate) {
                remaining /= candidate;
            }
        }
        candidate += 1;
    }
    remaining == 1 || gf_pow(2, order / remaining, polynomial, degree) != 1
}

fn gf_pow(mut base: u64, mut exponent: u64, polynomial: u64, degree: u32) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = gf_multiply(result, base, polynomial, degree);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = gf_multiply(base, base, polynomial, degree);
        }
    }
    result
}

fn gf_multiply(mut left: u64, mut right: u64, polynomial: u64, degree: u32) -> u64 {
    let mut result = 0;
    while right != 0 {
        if right & 1 != 0 {
            result ^= left;
        }
        right >>= 1;
        left <<= 1;
        if left & (1_u64 << degree) != 0 {
            left ^= polynomial;
        }
    }
    result & ((1_u64 << degree) - 1)
}

fn build_direction_numbers(
    degree: u32,
    polynomial: u32,
    initial: &[u32],
) -> [u64; MAX_BITS as usize] {
    let mut directions = [0_u64; MAX_BITS as usize];
    initial
        .iter()
        .copied()
        .enumerate()
        .for_each(|(index, value)| {
            directions[index] = u64::from(value) << (MAX_BITS as usize - 1 - index);
        });

    // Bratley-Fox recurrence in left-aligned form:
    // v_i = a_1 v_{i-1} ^ ... ^ a_{d-1} v_{i-d+1} ^ v_{i-d} ^ (v_{i-d} >> d),
    // where a_k are the encoded polynomial coefficients. Extending to all 64
    // bits lets the same table serve both f32 and f64 outputs.
    let degree_index = degree as usize;
    for index in degree_index..MAX_BITS as usize {
        directions[index] =
            directions[index - degree_index] ^ (directions[index - degree_index] >> degree);
        for coefficient in 1..degree_index {
            if (polynomial >> (degree - 1 - coefficient as u32)) & 1 == 1 {
                directions[index] ^= directions[index - coefficient];
            }
        }
    }
    directions
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn built_in_dimension_limits_match_preparation() {
        let plan = super::super::QmcPlan::sobol(JOE_KUO_DIMENSIONS).unwrap();
        assert_eq!(plan.dim(), JOE_KUO_DIMENSIONS);
        assert!(super::super::QmcPlan::sobol(JOE_KUO_DIMENSIONS + 1).is_err());
        assert_eq!(
            super::super::QmcPlan::niederreiter_base2(12).unwrap().dim(),
            12
        );
        assert!(super::super::QmcPlan::niederreiter_base2(13).is_err());
        assert!(super::super::QmcPlan::halton(100).is_ok());
        assert!(super::super::QmcPlan::halton(101).is_err());
    }

    #[test]
    fn every_bundled_definition_is_valid_and_dimension_ordered() {
        assert_eq!(JOE_KUO_D6.lines().count(), JOE_KUO_DIMENSIONS);
        for (index, line) in JOE_KUO_D6.lines().skip(1).enumerate() {
            let words: Vec<u32> = line
                .split_whitespace()
                .map(|word| word.parse().unwrap())
                .collect();
            assert_eq!(words[0] as usize, index + 2);
            let _ = SobolDirection::new(words[1], words[2], words[3..].to_vec()).unwrap();
        }
    }

    #[test]
    fn canonical_prefix_has_distinct_early_coordinates() {
        let directions = generate_direction_numbers(8, &DirectionNumberSet::JoeKuoD6).unwrap();
        assert!(directions.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(
            directions[1][..4],
            [
                0x8000_0000_0000_0000,
                0xc000_0000_0000_0000,
                0xa000_0000_0000_0000,
                0xf000_0000_0000_0000
            ]
        );
        assert_eq!(
            directions[2][..4],
            [
                0x8000_0000_0000_0000,
                0xc000_0000_0000_0000,
                0x6000_0000_0000_0000,
                0x9000_0000_0000_0000
            ]
        );
    }

    #[test]
    fn custom_definition_reproduces_canonical_dimension_three() {
        let custom = generate_direction_numbers(
            3,
            &DirectionNumberSet::Custom(vec![
                SobolDirection::new(1, 0, vec![1]).unwrap(),
                SobolDirection::new(2, 1, vec![1, 3]).unwrap(),
            ]),
        )
        .unwrap();
        let canonical = generate_direction_numbers(3, &DirectionNumberSet::JoeKuoD6).unwrap();
        assert_eq!(custom, canonical);
    }

    #[test]
    fn malformed_custom_definitions_are_rejected() {
        for result in [
            SobolDirection::new(0, 0, vec![]),
            SobolDirection::new(2, 0, vec![1, 3]),
            SobolDirection::new(2, 1, vec![1]),
            SobolDirection::new(2, 1, vec![1, 4]),
            SobolDirection::new(2, 1, vec![1, 5]),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        }
    }

    #[test]
    fn insufficient_custom_table_is_rejected() {
        let custom = vec![SobolDirection::new(1, 0, vec![1]).unwrap()];
        let error = generate_direction_numbers(5, &DirectionNumberSet::Custom(custom)).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
}
