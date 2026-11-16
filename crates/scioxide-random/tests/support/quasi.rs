//! Shared deterministic quasi-random fixtures for tests.

use scioxide_core::SciOxideError;
use scioxide_random::{DirectionNumberSet, SobolDirection};

// Dimensions 2 through 32 from Joe and Kuo's recommended D(6) table,
// `new-joe-kuo-6.21201` (16 September 2010).
const JOE_KUO_D6_32: &[(u32, u32, &[u32])] = &[
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 7, 11, 19]),
    (5, 11, &[1, 1, 5, 1, 1]),
    (5, 13, &[1, 1, 1, 3, 11]),
    (5, 14, &[1, 3, 5, 5, 31]),
    (6, 1, &[1, 3, 3, 9, 7, 49]),
    (6, 13, &[1, 1, 1, 15, 21, 21]),
    (6, 16, &[1, 3, 1, 13, 27, 49]),
    (6, 19, &[1, 1, 1, 15, 7, 5]),
    (6, 22, &[1, 3, 1, 15, 13, 25]),
    (6, 25, &[1, 1, 5, 5, 19, 61]),
    (7, 1, &[1, 3, 7, 11, 23, 15, 103]),
    (7, 4, &[1, 3, 7, 13, 13, 15, 69]),
    (7, 7, &[1, 1, 3, 13, 7, 35, 63]),
    (7, 8, &[1, 3, 5, 9, 1, 25, 53]),
    (7, 14, &[1, 3, 1, 13, 9, 35, 107]),
    (7, 19, &[1, 3, 1, 5, 27, 61, 31]),
    (7, 21, &[1, 1, 5, 11, 19, 41, 61]),
    (7, 28, &[1, 3, 5, 3, 3, 13, 69]),
    (7, 31, &[1, 1, 7, 13, 1, 19, 1]),
    (7, 32, &[1, 3, 7, 5, 13, 19, 59]),
    (7, 37, &[1, 1, 3, 9, 25, 29, 41]),
    (7, 41, &[1, 3, 5, 13, 23, 1, 55]),
    (7, 42, &[1, 3, 7, 3, 13, 59, 17]),
];

/// Return a fixed canonical custom direction table through dimension 32.
pub fn joe_kuo_d6_32() -> Result<DirectionNumberSet, SciOxideError> {
    Ok(DirectionNumberSet::Custom(
        JOE_KUO_D6_32
            .iter()
            .map(|&(degree, polynomial, initial)| {
                let mut values = scioxide_core::backend::try_with_capacity(
                    initial.len(),
                    scioxide_core::quantity!("sobol_direction_fixture"),
                )?;
                values.extend_from_slice(initial);
                SobolDirection::new(degree, polynomial, values)
            })
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
