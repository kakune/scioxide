//! Binary linear transformations; matrix rows are ordered most significant first.
use super::plan::Applied;
use crate::util::{SPLITMIX64_GAMMA, splitmix64_mix};
/// Prepared randomization words per coordinate: a 64-row matrix and its shift,
/// or one shift.
pub(super) const fn randomization_width(method: Applied) -> usize {
    if matches!(method, Applied::LinearMatrixShift) {
        65
    } else {
        1
    }
}

/// A unit lower-triangular binary matrix: keep the more significant bits and
/// force the diagonal. The final shift word and Halton shifts are unrestricted.
pub(super) const fn canonical_randomization_word(method: Applied, row: usize, word: u64) -> u64 {
    if matches!(method, Applied::LinearMatrixShift) && row < 64 {
        let diagonal = 1_u64 << (63 - row);
        (word & !(diagonal | (diagonal - 1))) | diagonal
    } else {
        word
    }
}

pub(super) fn transform(word: u64, dimension: usize, method: Applied, data: &[u64]) -> u64 {
    match method {
        // Random digital shift: XOR with a per-dimension word derived from the
        // seed. It preserves the net structure and makes each point uniformly
        // distributed.
        Applied::DigitalShift(seed) => {
            word ^ splitmix64_mix(seed ^ (dimension as u64).wrapping_mul(SPLITMIX64_GAMMA))
        }
        // Matousek linear matrix scramble plus shift. Each dimension owns 65
        // words: 64 row masks of a (lower-triangular, nonsingular) GF(2)
        // matrix followed by the shift word. Output bit `row` is the parity of
        // the input bits selected by that row's mask.
        Applied::LinearMatrixShift => {
            let matrix = &data[65 * dimension..65 * (dimension + 1)];
            matrix[..64]
                .iter()
                .enumerate()
                .fold(matrix[64], |value, (row, mask)| {
                    value ^ (u64::from((word & mask).count_ones() & 1) << (63 - row))
                })
        }
        _ => word,
    }
}
