//! Portable QMC checkpoints in the crate's little-endian, CRC-32C-checked format.
use super::plan::{Applied, QmcData};
use super::scramble;
use super::{
    DirectionNumberSet, QmcFamily, QmcPlan, QmcPosition, QmcScramble, QmcSequence, SobolDirection,
};
use crate::{SnapshotGenerator, state};
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SciOxideError};

/// Quasi-random plan and position decoded from a portable checkpoint.
///
/// Export a cursor with [`QmcSequence::export_state`] and decode it with
/// [`QmcPlan::import_state`]. The checkpoint does not depend on the output type, so a cursor
/// exported as `f64` can resume as `f32`. A cursor made from it continues with the same points.
///
/// # Examples
///
/// ```
/// use scioxide_random::{QmcPlan, QuasiRng};
/// let plan = QmcPlan::halton(2)?;
/// let mut sequence = plan.sequence::<f64>()?;
/// sequence.skip(5)?;
/// let bytes = sequence.export_state()?;
/// let checkpoint = QmcPlan::import_state(&bytes)?;
/// assert_eq!(checkpoint.position().next_index(), Some(5));
/// let mut restored = checkpoint.sequence::<f64>()?;
/// assert_eq!(restored.next_point()?, sequence.next_point()?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct QmcCheckpoint {
    plan: QmcPlan,
    position: QmcPosition,
}
impl QmcCheckpoint {
    /// Returns the plan stored in the checkpoint.
    pub const fn plan(&self) -> &QmcPlan {
        &self.plan
    }
    /// Returns the sequence position stored in the checkpoint.
    pub const fn position(&self) -> QmcPosition {
        self.position
    }
    /// Creates a cursor at the stored position with output type `S`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the cursor's working storage cannot be allocated.
    pub fn sequence<S: RealScalar>(&self) -> Result<QmcSequence<'_, S>, SciOxideError> {
        let mut cursor = self.plan.sequence()?;
        cursor.restore_snapshot(self.position);
        Ok(cursor)
    }
}
impl<S: RealScalar> QmcSequence<'_, S> {
    /// Returns the size in bytes of this cursor's checkpoint, including plan data, header and
    /// checksum.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the plan data does not fit the checkpoint format's 32-bit length.
    pub fn state_bytes(&self) -> Result<usize, SciOxideError> {
        encoded_len(self.plan)
    }
    /// Returns a portable checkpoint of the plan and position, without advancing the cursor.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::state_bytes`].
    /// - `AllocationFailure` if the buffer cannot be allocated.
    pub fn export_state(&self) -> Result<Vec<u8>, SciOxideError> {
        let mut bytes = scioxide_core::backend::try_filled(
            self.state_bytes()?,
            0,
            scioxide_core::quantity!("qmc_checkpoint"),
        )?;
        self.export_state_into(&mut bytes)?;
        Ok(bytes)
    }
    /// Writes the checkpoint into `output`, which must have exactly [`Self::state_bytes`] bytes.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` has the wrong length.
    /// - The errors of [`Self::state_bytes`].
    ///
    /// On error, `output` is unchanged.
    pub fn export_state_into(&self, output: &mut [u8]) -> Result<(), SciOxideError> {
        let len = self.state_bytes()?;
        let (tag, seed) = self.plan.scramble.tag();
        state::write_envelope(
            output,
            self.plan.family as u8,
            algorithm_version(tag),
            len - state::ENVELOPE_BYTES,
            u8::from(self.position.exhausted),
            |output| {
                let mut offset = 0;
                let mut write = |bytes: &[u8]| {
                    output[offset..offset + bytes.len()].copy_from_slice(bytes);
                    offset += bytes.len();
                };
                write(&(self.plan.dim() as u32).to_le_bytes());
                write(&self.position.index.to_le_bytes());
                write(&[tag]);
                write(&seed.to_le_bytes());
                // Only the defining inputs are serialized (polynomials, initial
                // direction numbers, bases, permutations and randomization words);
                // derived direction tables are rebuilt on import, so a checkpoint
                // cannot smuggle in an inconsistent generator matrix.
                match &self.plan.data {
                    QmcData::Sobol { source, .. } => {
                        if let DirectionNumberSet::Custom(definitions) = source {
                            write(&[1]);
                            for definition in definitions {
                                write(&definition.degree().to_le_bytes());
                                write(&definition.primitive_polynomial().to_le_bytes());
                                for value in definition.initial_direction_numbers() {
                                    write(&value.to_le_bytes());
                                }
                            }
                        } else {
                            write(&[0]);
                        }
                    }
                    QmcData::Niederreiter { .. } => write(&[0]),
                    QmcData::Radical {
                        bases,
                        permutations,
                        ..
                    } => {
                        write(&[0]);
                        for base in bases {
                            write(&base.to_le_bytes());
                        }
                        if tag == 5 {
                            for permutation in permutations {
                                for value in permutation {
                                    write(&value.to_le_bytes());
                                }
                            }
                        }
                    }
                }
                for word in &self.plan.randomization {
                    write(&word.to_le_bytes());
                }
            },
        )
    }
}
fn encoded_len(plan: &QmcPlan) -> Result<usize, SciOxideError> {
    let extra = match &plan.data {
        QmcData::Sobol {
            source: DirectionNumberSet::Custom(definitions),
            ..
        } => definitions.iter().try_fold(0_usize, |sum, def| {
            sum.checked_add(8 + 4 * def.initial_direction_numbers().len())
        }),
        QmcData::Sobol { .. } | QmcData::Niederreiter { .. } => Some(0),
        QmcData::Radical {
            bases,
            permutations,
            ..
        } => {
            let base_bytes = bases.len().checked_mul(4);
            if plan.scramble == Applied::CustomPermutation {
                permutations
                    .iter()
                    .try_fold(base_bytes.unwrap_or(usize::MAX), |sum, p| {
                        sum.checked_add(p.len().checked_mul(4)?)
                    })
            } else {
                base_bytes
            }
        }
    }
    .ok_or_else(|| {
        state::malformed(
            "QMC checkpoint size overflow",
            scioxide_core::argument!("self"),
        )
    })?;
    let payload = extra
        .checked_add(plan.randomization.len().checked_mul(8).ok_or_else(|| {
            state::malformed(
                "randomization size overflow",
                scioxide_core::argument!("self"),
            )
        })?)
        .and_then(|n| n.checked_add(22))
        .ok_or_else(|| {
            state::malformed(
                "QMC checkpoint size overflow",
                scioxide_core::argument!("self"),
            )
        })?;
    if u32::try_from(payload).is_err() || u32::try_from(plan.dim()).is_err() {
        return Err(state::malformed(
            "QMC checkpoint exceeds schema lengths",
            scioxide_core::argument!("self"),
        ));
    }
    payload.checked_add(state::ENVELOPE_BYTES).ok_or_else(|| {
        state::malformed(
            "QMC checkpoint size overflow",
            scioxide_core::argument!("self"),
        )
    })
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SciOxideError> {
        let bytes = state::read(self.bytes, self.offset, "QMC payload")?;
        self.offset += N;
        Ok(bytes)
    }
    fn word(&mut self) -> Result<u32, SciOxideError> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn words(&mut self, count: usize) -> Result<Vec<u32>, SciOxideError> {
        if count > self.bytes.len().saturating_sub(self.offset) / 4 {
            return Err(state::malformed(
                "truncated QMC word array",
                scioxide_core::argument!("bytes"),
            ));
        }
        let mut result = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("qmc_checkpoint_words"),
        )?;
        for _ in 0..count {
            result.push(self.word()?);
        }
        Ok(result)
    }
}
// Only payload validation is reclassified; resource and numerical failures retain
// their original structured errors. Envelope versions and restore identity are
// checked outside this boundary.
fn payload_validation(error: SciOxideError) -> SciOxideError {
    match error.kind() {
        scioxide_core::SciOxideErrorKind::InvalidInput
        | scioxide_core::SciOxideErrorKind::DomainError
        | scioxide_core::SciOxideErrorKind::DimensionMismatch => state::malformed(
            "invalid QMC checkpoint payload",
            scioxide_core::argument!("bytes"),
        ),
        _ => error,
    }
}

impl QmcPlan {
    /// Decodes a checkpoint written by [`QmcSequence::export_state`] into a plan and position.
    ///
    /// The output type is chosen when you create a cursor with [`QmcCheckpoint::sequence`].
    /// Lengths in the checkpoint are checked before any array is allocated, and the plan is
    /// validated as if it were constructed directly.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Malformed`) if the checkpoint is truncated, has a wrong length or
    ///   checksum, has an invalid flag or seed, or describes an invalid plan.
    /// - `UnsupportedConfiguration` for an unknown format, algorithm version, family tag,
    ///   scramble tag or endian marker. Seeded digit permutations need algorithm version 2 and all
    ///   other scrambles version 1, and a checkpoint with the other version is rejected.
    /// - `AllocationFailure` if the plan's storage cannot be allocated.
    pub fn import_state(bytes: &[u8]) -> Result<QmcCheckpoint, SciOxideError> {
        let tag = *bytes.get(6).ok_or_else(|| {
            state::malformed("truncated QMC header", scioxide_core::argument!("bytes"))
        })?;
        let family = match tag {
            32 => QmcFamily::Sobol,
            33 => QmcFamily::Halton,
            34 => QmcFamily::ReverseHalton,
            35 => QmcFamily::NiederreiterBase2,
            _ => {
                return Err(state::unsupported(
                    "unknown QMC algorithm",
                    scioxide_core::argument!("bytes"),
                ));
            }
        };
        let flags = state::validate_envelope(bytes, tag, 1..=2, None, 1)?;
        let mut reader = Reader {
            bytes: &bytes[..bytes.len() - 4],
            offset: state::PREFIX_BYTES,
        };
        let dim = reader.word()? as usize;
        let index = u64::from_le_bytes(reader.take()?);
        if flags == 1 && index != u64::MAX {
            return Err(state::malformed(
                "exhausted QMC index must be u64::MAX",
                scioxide_core::argument!("bytes"),
            ));
        }
        let method = reader.take::<1>()?[0];
        let seed = u64::from_le_bytes(reader.take()?);
        let scramble = Applied::from_tag(method, seed).ok_or_else(|| {
            state::unsupported(
                "unknown QMC scramble tag",
                scioxide_core::argument!("bytes"),
            )
        })?;
        // Seeded digit permutations changed convention in QMC algorithm
        // version 2; an otherwise valid earlier checkpoint is unsupported
        // rather than restored as a different stream.
        let version_supported = bytes[7] == algorithm_version(method);
        if !matches!(method, 1 | 4) && seed != 0 {
            return Err(state::malformed(
                "noncanonical QMC seed",
                scioxide_core::argument!("bytes"),
            ));
        }
        let source = reader.take::<1>()?[0];
        if dim == 0 || source > 1 || (family != QmcFamily::Sobol && source != 0) {
            return Err(state::malformed(
                "invalid QMC dimension or source",
                scioxide_core::argument!("bytes"),
            ));
        }
        let mut plan = match family {
            QmcFamily::Sobol if source == 0 => Self::sobol(dim).map_err(payload_validation)?,
            QmcFamily::Sobol => {
                if dim - 1 > reader.bytes.len().saturating_sub(reader.offset) / 12 {
                    return Err(state::malformed(
                        "truncated Sobol definitions",
                        scioxide_core::argument!("bytes"),
                    ));
                }
                let mut definitions = scioxide_core::backend::try_with_capacity(
                    dim - 1,
                    scioxide_core::quantity!("qmc_custom_directions"),
                )?;
                for _ in 1..dim {
                    let degree = reader.word()?;
                    let polynomial = reader.word()?;
                    if !(1..=32).contains(&degree) {
                        return Err(state::malformed(
                            "invalid Sobol degree",
                            scioxide_core::argument!("bytes"),
                        ));
                    }
                    definitions.push(
                        SobolDirection::new(degree, polynomial, reader.words(degree as usize)?)
                            .map_err(payload_validation)?,
                    );
                }
                Self::sobol_with_direction_numbers(dim, DirectionNumberSet::Custom(definitions))
                    .map_err(payload_validation)?
            }
            QmcFamily::NiederreiterBase2 => {
                Self::niederreiter_base2(dim).map_err(payload_validation)?
            }
            QmcFamily::Halton => {
                Self::halton_with_bases(reader.words(dim)?).map_err(payload_validation)?
            }
            QmcFamily::ReverseHalton => {
                Self::reverse_halton_with_bases(reader.words(dim)?).map_err(payload_validation)?
            }
        };
        // The plan's one compatibility rule also guards decoded checkpoints.
        if !super::plan::compatible(family, scramble) {
            return Err(state::malformed(
                "scramble is incompatible with the QMC family",
                scioxide_core::argument!("bytes"),
            ));
        }
        match scramble {
            Applied::None => {}
            Applied::DigitalShift(seed) => {
                plan = plan
                    .with_scramble(QmcScramble::DigitalShift(seed))
                    .map_err(payload_validation)?;
            }
            Applied::DigitPermutation(seed) => {
                plan = plan
                    .with_scramble(QmcScramble::DigitPermutation(seed))
                    .map_err(payload_validation)?;
            }
            Applied::CustomPermutation => {
                let bases = plan.bases().unwrap_or_default();
                let mut permutations = scioxide_core::backend::try_with_capacity(
                    dim,
                    scioxide_core::quantity!("qmc_permutations"),
                )?;
                for &base in bases {
                    permutations.push(reader.words(base as usize)?);
                }
                plan = plan
                    .with_permutations(permutations)
                    .map_err(payload_validation)?;
            }
            Applied::LinearMatrixShift | Applied::RandomShift => {
                let width = scramble::randomization_width(scramble);
                let len = dim.checked_mul(width).ok_or_else(|| {
                    state::malformed("randomization size", scioxide_core::argument!("bytes"))
                })?;
                if len > reader.bytes.len().saturating_sub(reader.offset) / 8 {
                    return Err(state::malformed(
                        "truncated randomization",
                        scioxide_core::argument!("bytes"),
                    ));
                }
                let mut data = scioxide_core::backend::try_with_capacity(
                    len,
                    scioxide_core::quantity!("randomization_import"),
                )?;
                for i in 0..len {
                    let word = u64::from_le_bytes(reader.take()?);
                    if word != scramble::canonical_randomization_word(scramble, i % width, word) {
                        return Err(state::malformed(
                            "invalid triangular scramble",
                            scioxide_core::argument!("bytes"),
                        ));
                    }
                    data.push(word);
                }
                plan.scramble = scramble;
                plan.randomization = data;
                plan.prepare_digital().map_err(payload_validation)?;
            }
        }
        if reader.offset != reader.bytes.len() {
            return Err(state::malformed(
                "trailing QMC payload",
                scioxide_core::argument!("bytes"),
            ));
        }
        if !version_supported {
            return Err(state::unsupported(
                "QMC algorithm version does not match the scramble convention",
                scioxide_core::argument!("bytes"),
            ));
        }
        Ok(QmcCheckpoint {
            plan,
            position: QmcPosition {
                index,
                exhausted: flags == 1,
            },
        })
    }
}

/// QMC algorithm version of a checkpointed scramble convention: seeded digit
/// permutations (tag 4) use the shared bounded index draw from version 2.
const fn algorithm_version(scramble_tag: u8) -> u8 {
    if scramble_tag == 4 { 2 } else { 1 }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{QmcRandomization, QuasiRng, SeedableRng, SplitMix64};
    use scioxide_core::{InvalidInputReason, SciOxideErrorView};

    fn reseal(bytes: &mut [u8]) {
        let end = bytes.len() - 4;
        let checksum = state::crc32c(&bytes[..end]);
        bytes[end..].copy_from_slice(&checksum.to_le_bytes());
    }

    fn malformed(bytes: &[u8]) {
        crate::error_assertions::invalid(
            QmcPlan::import_state(bytes).unwrap_err(),
            InvalidInputReason::Malformed,
        );
    }

    #[test]
    fn payload_constructor_failures_are_malformed_for_every_family() {
        for (plan, offset, value) in [
            (QmcPlan::sobol(2).unwrap(), 16, 21_202_u32),
            (QmcPlan::niederreiter_base2(2).unwrap(), 16, 13),
            (QmcPlan::halton(2).unwrap(), 38, 4),
            (QmcPlan::reverse_halton(2).unwrap(), 38, 4),
        ] {
            let mut sequence = plan.sequence::<f64>().unwrap();
            sequence.skip(7).unwrap();
            let original = sequence.export_state().unwrap();
            let mut bytes = original.clone();
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            reseal(&mut bytes);
            malformed(&bytes);
            assert_eq!(sequence.export_state().unwrap(), original);
        }
    }

    #[test]
    fn payload_directions_permutations_and_scrambles_are_malformed() {
        let custom = QmcPlan::sobol_with_direction_numbers(
            2,
            DirectionNumberSet::Custom(vec![SobolDirection::new(1, 0, vec![1]).unwrap()]),
        )
        .unwrap();
        let mut bytes = custom.sequence::<f64>().unwrap().export_state().unwrap();
        bytes[46..50].copy_from_slice(&2_u32.to_le_bytes()); // Even m_1.
        reseal(&mut bytes);
        malformed(&bytes);
        let custom = QmcPlan::halton(2)
            .unwrap()
            .with_permutations(vec![vec![0, 1], vec![0, 2, 1]])
            .unwrap();
        let mut bytes = custom.sequence::<f64>().unwrap().export_state().unwrap();
        bytes[50..54].copy_from_slice(&0_u32.to_le_bytes()); // Duplicate digit.
        reseal(&mut bytes);
        malformed(&bytes);
        for (plan, method) in [
            (QmcPlan::sobol(2).unwrap(), 4), // Digit permutations require Halton.
            (QmcPlan::niederreiter_base2(2).unwrap(), 4),
            (QmcPlan::halton(2).unwrap(), 1), // Digital shifts require base 2.
            (QmcPlan::reverse_halton(2).unwrap(), 1),
        ] {
            let mut bytes = plan.sequence::<f64>().unwrap().export_state().unwrap();
            bytes[28] = method;
            reseal(&mut bytes);
            malformed(&bytes);
        }
    }

    #[test]
    fn payload_validation_preserves_resource_failures() {
        let error =
            SciOxideError::allocation_failure(scioxide_core::quantity!("qmc_checkpoint"), 64);
        assert_eq!(
            payload_validation(error).kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
    }

    #[test]
    fn all_scrambles_have_a_compact_canonical_payload() {
        let mut rng = SplitMix64::from_seed(42);
        let plans = [
            QmcPlan::sobol(2).unwrap(),
            QmcPlan::sobol(2)
                .unwrap()
                .with_scramble(QmcScramble::DigitalShift(11))
                .unwrap(),
            QmcPlan::sobol(2)
                .unwrap()
                .with_randomization(QmcRandomization::LinearMatrixShift, &mut rng)
                .unwrap(),
            QmcPlan::halton(2)
                .unwrap()
                .with_randomization(QmcRandomization::RandomShift, &mut rng)
                .unwrap(),
            QmcPlan::halton(2)
                .unwrap()
                .with_scramble(QmcScramble::DigitPermutation(11))
                .unwrap(),
            QmcPlan::halton(2)
                .unwrap()
                .with_permutations(vec![vec![0, 1], vec![0, 2, 1]])
                .unwrap(),
        ];
        for (tag, plan) in plans.into_iter().enumerate() {
            let mut sequence = plan.sequence::<f64>().unwrap();
            sequence.skip(7).unwrap();
            let original = sequence.export_state().unwrap();
            assert_eq!(original[28], tag as u8);
            assert_eq!(original[37], 0); // Built-in direction source.
            let extra = match tag {
                2 => 2 * 65 * 8,          // Digital matrix and shift words.
                3 => 2 * 4 + 2 * 8,       // Halton bases and random shifts.
                4 => 2 * 4,               // Halton bases.
                5 => 2 * 4 + (2 + 3) * 4, // Bases and custom permutations.
                _ => 0,
            };
            assert_eq!(original.len(), state::ENVELOPE_BYTES + 22 + extra);
            if !matches!(tag, 1 | 4) {
                assert_eq!(&original[29..37], &[0; 8]);
                for offset in 29..37 {
                    let mut bytes = original.clone();
                    bytes[offset] = 1;
                    reseal(&mut bytes);
                    malformed(&bytes);
                }
            } else {
                assert_eq!(u64::from_le_bytes(original[29..37].try_into().unwrap()), 11);
            }
            // Only seeded digit permutations use QMC algorithm version 2; the
            // other version is unsupported rather than a different stream.
            assert_eq!(original[7], if tag == 4 { 2 } else { 1 });
            let mut other = original.clone();
            other[7] = if tag == 4 { 1 } else { 2 };
            reseal(&mut other);
            assert!(matches!(
                QmcPlan::import_state(&other).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
            let restored = QmcPlan::import_state(&original).unwrap();
            assert_eq!(
                restored.sequence::<f32>().unwrap().export_state().unwrap(),
                original
            );
            let mut resumed = restored.sequence::<f64>().unwrap();
            for _ in 0..3 {
                assert_eq!(
                    resumed.next_point().unwrap(),
                    sequence.next_point().unwrap()
                );
            }
        }
    }

    #[test]
    fn unsupported_scramble_tags_are_rejected() {
        let original = QmcPlan::sobol(1)
            .unwrap()
            .sequence::<f64>()
            .unwrap()
            .export_state()
            .unwrap();
        for tag in [6, 7, 8, 255] {
            let mut bytes = original.clone();
            bytes[28] = tag;
            reseal(&mut bytes);
            assert!(matches!(
                QmcPlan::import_state(&bytes).unwrap_err().view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
        }
    }

    #[test]
    fn imported_matrix_requires_every_diagonal_and_forbids_bits_below_it() {
        for base in [
            QmcPlan::sobol(2).unwrap(),
            QmcPlan::niederreiter_base2(2).unwrap(),
        ] {
            let plan = base
                .with_randomization(
                    QmcRandomization::LinearMatrixShift,
                    &mut SplitMix64::from_seed(42),
                )
                .unwrap();
            let original = plan.sequence::<f64>().unwrap().export_state().unwrap();
            for dimension in 0..2 {
                for row in 0..64 {
                    let offset = 38 + 8 * (65 * dimension + row);
                    let diagonal = 1_u64 << (63 - row);
                    let word = u64::from_le_bytes(original[offset..offset + 8].try_into().unwrap());
                    assert_eq!(word & diagonal, diagonal);
                    assert_eq!(word & (diagonal - 1), 0);
                    let mut bytes = original.clone();
                    bytes[offset..offset + 8].copy_from_slice(&(word ^ diagonal).to_le_bytes());
                    reseal(&mut bytes);
                    malformed(&bytes);
                    if row < 63 {
                        let mut bytes = original.clone();
                        bytes[offset..offset + 8]
                            .copy_from_slice(&(word | (diagonal >> 1)).to_le_bytes());
                        reseal(&mut bytes);
                        malformed(&bytes);
                    }
                }
                // The 65th word is a shift, so every bit pattern is canonical.
                for shift in [0_u64, u64::MAX] {
                    let offset = 38 + 8 * (65 * dimension + 64);
                    let mut bytes = original.clone();
                    bytes[offset..offset + 8].copy_from_slice(&shift.to_le_bytes());
                    reseal(&mut bytes);
                    let checkpoint = QmcPlan::import_state(&bytes).unwrap();
                    assert_eq!(
                        checkpoint
                            .sequence::<f64>()
                            .unwrap()
                            .export_state()
                            .unwrap(),
                        bytes
                    );
                }
            }
        }
    }
}
