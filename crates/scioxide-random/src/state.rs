//! Version-1 generator checkpoints. Parsing has no side effects; the generator
//! changes only after the whole checkpoint is validated.

use crate::{BitGenerator, SerializableState};
use scioxide_core::{InvalidInputReason, SciOxideError};

const MAGIC: [u8; 4] = *b"GRN2";
pub(crate) const PREFIX_BYTES: usize = 16;
const TRAILER_BYTES: usize = 4;
pub(crate) const ENVELOPE_BYTES: usize = PREFIX_BYTES + TRAILER_BYTES;

/// Identifies a built-in generator algorithm; the discriminant is its tag in checkpoints.
///
/// ```
/// use scioxide_random::{BitGenerator, GeneratorAlgorithm, Mt19937, SeedableRng};
/// let rng = Mt19937::from_seed(5489);
/// let algorithm = rng.stream_identity().map(|identity| identity.algorithm());
/// assert_eq!(algorithm, Some(GeneratorAlgorithm::Mt19937));
/// assert_eq!(GeneratorAlgorithm::Mt19937 as u8, 7); // checkpoint tag
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GeneratorAlgorithm {
    /// [`SplitMix64`](crate::SplitMix64).
    SplitMix64 = 1,
    /// [`Xoshiro256StarStar`](crate::Xoshiro256StarStar), with the jump polynomials of Blackman and Vigna.
    Xoshiro256StarStar = 2,
    /// [`Xoshiro128StarStar`](crate::Xoshiro128StarStar), with the jump polynomials of Blackman and Vigna.
    Xoshiro128StarStar = 3,
    /// [`Pcg64`](crate::Pcg64) (PCG-XSL-RR); the state advances before each output.
    Pcg64 = 4,
    /// [`Lcg64`](crate::Lcg64), a full-period 64-bit linear congruential generator.
    Lcg64 = 5,
    /// [`Mrg32k3a`](crate::Mrg32k3a), the integer form of the recurrence.
    Mrg32k3a = 6,
    /// [`Mt19937`](crate::Mt19937).
    Mt19937 = 7,
    /// [`Ranlux`](crate::Ranlux).
    Ranlux = 8,
    /// [`Ranlxs`](crate::Ranlxs).
    Ranlxs = 9,
    /// [`Ranlxd`](crate::Ranlxd).
    Ranlxd = 10,
    /// [`Taus2`](crate::Taus2).
    Taus2 = 11,
    /// [`Taus113`](crate::Taus113).
    Taus113 = 12,
    /// [`Gfsr4`](crate::Gfsr4).
    Gfsr4 = 13,
    /// [`Mrg`](crate::Mrg).
    Mrg = 14,
    /// [`Cmrg`](crate::Cmrg).
    Cmrg = 15,
    /// [`Philox4x64`](crate::Philox4x64), ten rounds.
    Philox4x64 = 16,
    /// [`Threefry4x64`](crate::Threefry4x64), twenty rounds.
    Threefry4x64 = 17,
}

impl GeneratorAlgorithm {
    pub(crate) fn from_tag(tag: u8) -> Result<Self, SciOxideError> {
        match tag {
            1 => Ok(Self::SplitMix64),
            2 => Ok(Self::Xoshiro256StarStar),
            3 => Ok(Self::Xoshiro128StarStar),
            4 => Ok(Self::Pcg64),
            5 => Ok(Self::Lcg64),
            6 => Ok(Self::Mrg32k3a),
            7 => Ok(Self::Mt19937),
            8 => Ok(Self::Ranlux),
            9 => Ok(Self::Ranlxs),
            10 => Ok(Self::Ranlxd),
            11 => Ok(Self::Taus2),
            12 => Ok(Self::Taus113),
            13 => Ok(Self::Gfsr4),
            14 => Ok(Self::Mrg),
            15 => Ok(Self::Cmrg),
            16 => Ok(Self::Philox4x64),
            17 => Ok(Self::Threefry4x64),
            _ => Err(unsupported(
                "unknown random generator algorithm",
                scioxide_core::argument!("bytes"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeOrigin<'a> {
    Wide(&'a [u64]),
    Narrow(&'a [u32]),
}

/// Identity of the sequence a built-in generator produces, independent of its position.
///
/// Two identities are equal if they have the same algorithm, the same root
/// state and the same stream selectors. The comparison is exact; it uses no
/// hash. The position within the sequence is not part of the identity, so a
/// generator keeps its identity as it draws. Clones and restored checkpoints
/// have the identity of the original. An identity is read-only and cannot
/// reseed a generator.
///
/// ```
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng, SplittableRng};
/// let mut root = Pcg64::from_seed(1);
/// let start = root.clone();
/// root.next_u64()?;
/// assert_eq!(root.stream_identity(), start.stream_identity()); // position is not identity
/// let stream = root.split(3)?;
/// assert_ne!(stream.stream_identity(), root.stream_identity());
/// assert_eq!(stream.stream_identity().and_then(|id| id.selectors()), Some((3, 0)));
/// assert_eq!(root.stream_identity().and_then(|id| id.selectors()), None);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamIdentity<'a> {
    pub(crate) algorithm: GeneratorAlgorithm,
    pub(crate) origin: [u64; 6],
    pub(crate) selectors: Option<(u64, u64)>,
    pub(crate) extended_origin: NativeOrigin<'a>,
}
impl StreamIdentity<'_> {
    pub(crate) const fn new(
        algorithm: GeneratorAlgorithm,
        origin: [u64; 6],
        selectors: Option<(u64, u64)>,
    ) -> Self {
        Self {
            algorithm,
            origin,
            selectors,
            extended_origin: NativeOrigin::Wide(&[]),
        }
    }
    /// Returns the generator algorithm.
    pub const fn algorithm(self) -> GeneratorAlgorithm {
        self.algorithm
    }
    /// Returns the stream and substream indices, or `None` for a root that selected none.
    ///
    /// The root of [`Mrg32k3a`](crate::Mrg32k3a) has selectors `(0, 0)`. A
    /// [`Pcg64`](crate::Pcg64) with the default increment has none.
    pub const fn selectors(self) -> Option<(u64, u64)> {
        self.selectors
    }
}

pub(crate) struct Snapshot {
    pub identity: StreamIdentity<'static>,
    pub words: [u64; 6],
    pub parameter: u128,
    pub counter: Option<u128>,
}

pub(crate) trait StateEngine: BitGenerator + Sized {
    const ALGORITHM: GeneratorAlgorithm;
    const WORDS: usize;
    const WORD_BYTES: usize = 8;
    const PARAMETER_BYTES: usize = 0;
    const HAS_STREAMS: bool = false;
    const PAYLOAD_BYTES: usize = 2 * Self::WORDS * Self::WORD_BYTES
        + Self::PARAMETER_BYTES
        + if Self::HAS_STREAMS { 16 } else { 0 }
        + 16;
    fn snapshot(&self) -> Snapshot;
    fn from_snapshot(snapshot: Snapshot) -> Result<Self, SciOxideError>;
}
// One blanket implementation serves every snapshot engine.
impl<B: StateEngine> SerializableState for B {
    const STATE_BYTES: usize = ENVELOPE_BYTES + B::PAYLOAD_BYTES;
    fn export_state_into(&self, output: &mut [u8]) -> Result<(), SciOxideError> {
        encode_into::<B>(&self.snapshot(), output)
    }
    fn import_state(bytes: &[u8]) -> Result<Self, SciOxideError> {
        Self::from_snapshot(decode::<B>(bytes)?)
    }
}

#[inline]
fn encode_into<B: StateEngine>(
    snapshot: &Snapshot,
    output: &mut [u8],
) -> Result<(), SciOxideError> {
    let algorithm = snapshot.identity.algorithm;
    write_envelope(
        output,
        algorithm as u8,
        1,
        B::PAYLOAD_BYTES,
        u8::from(snapshot.counter.is_some())
            | (u8::from(snapshot.identity.selectors.is_some()) << 1),
        |output| {
            // Flag bit 0: position tracked; bit 1: stream selectors present. Absent
            // values are still written as zeros so the layout stays fixed-size and
            // `decode` can insist on a single canonical encoding.
            let mut offset = 0;
            for words in [&snapshot.identity.origin, &snapshot.words] {
                words.iter().take(B::WORDS).for_each(|word| {
                    let bytes = word.to_le_bytes();
                    let width = B::WORD_BYTES;
                    output[offset..offset + width].copy_from_slice(&bytes[..width]);
                    offset += width;
                });
            }
            if B::PARAMETER_BYTES != 0 {
                output[offset..offset + 16].copy_from_slice(&snapshot.parameter.to_le_bytes());
                offset += 16;
            }
            if B::HAS_STREAMS {
                let (stream, subsequence) = snapshot.identity.selectors.unwrap_or((0, 0));
                output[offset..offset + 8].copy_from_slice(&stream.to_le_bytes());
                output[offset + 8..offset + 16].copy_from_slice(&subsequence.to_le_bytes());
                offset += 16;
            }
            output[offset..offset + 16]
                .copy_from_slice(&snapshot.counter.unwrap_or(0).to_le_bytes());
        },
    )
}

#[inline]
fn decode<B: StateEngine>(bytes: &[u8]) -> Result<Snapshot, SciOxideError> {
    let algorithm = B::ALGORITHM;
    let flags = validate_envelope(bytes, algorithm as u8, 1..=1, Some(B::PAYLOAD_BYTES), 3)?;
    let mut offset = PREFIX_BYTES;
    let mut read_words = || -> Result<[u64; 6], SciOxideError> {
        let mut words = [0; 6];
        for word in words.iter_mut().take(B::WORDS) {
            *word = if B::WORD_BYTES == 4 {
                u64::from(u32::from_le_bytes(read(bytes, offset, "word")?))
            } else {
                u64::from_le_bytes(read(bytes, offset, "word")?)
            };
            offset += B::WORD_BYTES;
        }
        Ok(words)
    };
    let origin = read_words()?;
    let words = read_words()?;
    let parameter = if B::PARAMETER_BYTES != 0 {
        let value = u128::from_le_bytes(read(bytes, offset, "parameter")?);
        offset += 16;
        value
    } else {
        0
    };
    let selectors = if B::HAS_STREAMS {
        let stream = u64::from_le_bytes(read(bytes, offset, "stream")?);
        let subsequence = u64::from_le_bytes(read(bytes, offset + 8, "subsequence")?);
        offset += 16;
        // MRG32k3a always carries selectors (its root is `(0, 0)`), whereas
        // PCG uses `None` for the default increment; anything else would give
        // one state two encodings.
        if flags & 2 != 0 {
            Some((stream, subsequence))
        } else if stream == 0 && subsequence == 0 && algorithm != GeneratorAlgorithm::Mrg32k3a {
            None
        } else {
            return Err(malformed(
                "noncanonical stream selectors",
                scioxide_core::argument!("bytes"),
            ));
        }
    } else if flags & 2 != 0 {
        return Err(malformed(
            "algorithm has no stream selectors",
            scioxide_core::argument!("bytes"),
        ));
    } else {
        None
    };
    let count = u128::from_le_bytes(read(bytes, offset, "counter")?);
    let counter = if flags & 1 != 0 {
        Some(count)
    } else if count == 0 {
        None
    } else {
        return Err(malformed(
            "untracked counter must encode zero",
            scioxide_core::argument!("bytes"),
        ));
    };
    Ok(Snapshot {
        identity: StreamIdentity::new(algorithm, origin, selectors),
        words,
        parameter,
        counter,
    })
}

pub(crate) fn read<const N: usize>(
    bytes: &[u8],
    offset: usize,
    field: &'static str,
) -> Result<[u8; N], SciOxideError> {
    offset
        .checked_add(N)
        .and_then(|end| bytes.get(offset..end))
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| {
            malformed(
                alloc::format!("truncated checkpoint field: {field}"),
                scioxide_core::argument!("bytes"),
            )
        })
}
/// Malformed checkpoint error against `argument`.
pub(crate) fn malformed(
    detail: impl Into<alloc::borrow::Cow<'static, str>>,
    argument: scioxide_core::Argument,
) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Malformed, detail)
}
/// Unsupported checkpoint error against `argument`.
pub(crate) fn unsupported(
    detail: &'static str,
    argument: scioxide_core::Argument,
) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        InvalidInputReason::UnsupportedConfiguration,
        detail,
    )
}

// Reflected CRC-32C (Castagnoli, polynomial 0x82F63B78) table built at compile
// time. CRC-32C is used only to detect accidental corruption of checkpoints,
// not tampering.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = (value >> 1) ^ if value & 1 != 0 { 0x82f6_3b78 } else { 0 };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
};
pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        #[allow(clippy::cast_possible_truncation)]
        let index = usize::from((crc as u8) ^ byte);
        (crc >> 8) ^ CRC_TABLE[index]
    })
}

/// Shared variable/fixed payload envelope validation; no allocations or mutation.
pub(crate) fn validate_envelope(
    bytes: &[u8],
    expected: u8,
    versions: core::ops::RangeInclusive<u8>,
    payload_bytes: Option<usize>,
    allowed_flags: u8,
) -> Result<u8, SciOxideError> {
    let header = bytes.get(..PREFIX_BYTES).ok_or_else(|| {
        malformed(
            "truncated checkpoint header",
            scioxide_core::argument!("bytes"),
        )
    })?;
    if header[..4] != MAGIC {
        return Err(malformed(
            "invalid checkpoint magic",
            scioxide_core::argument!("bytes"),
        ));
    }
    // Header: magic[0..4], format version u16 [4..6], algorithm tag [6],
    // algorithm version [7], little-endian marker [8], flags [9], reserved
    // [10..12], payload length u32 [12..16].
    if header[4..6] != 1_u16.to_le_bytes() || !versions.contains(&header[7]) || header[8] != 1 {
        return Err(unsupported(
            "unsupported checkpoint format, algorithm version, or endian marker",
            scioxide_core::argument!("bytes"),
        ));
    }
    if header[6] != expected {
        // Engine tags live below 32 and QMC tags from 32 up. For an engine import,
        // an unknown tag is reported as unsupported rather than as a mismatch
        // so newer checkpoints are diagnosed correctly.
        if expected < 32 {
            GeneratorAlgorithm::from_tag(header[6])?;
        }
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("bytes"),
            InvalidInputReason::Inconsistent,
            "checkpoint algorithm does not match engine",
        ));
    }
    let payload_len = u32::from_le_bytes(read(header, 12, "payload length")?) as usize;
    if payload_bytes.is_some_and(|expected| expected != payload_len)
        || payload_len.checked_add(PREFIX_BYTES + TRAILER_BYTES) != Some(bytes.len())
    {
        return Err(malformed(
            "checkpoint length must match its exact algorithm layout",
            scioxide_core::argument!("bytes"),
        ));
    }
    // The checksum is verified before flags are inspected, so a corrupted
    // record is reported as corruption rather than as a semantic error.
    let end = bytes.len() - TRAILER_BYTES;
    if crc32c(&bytes[..end]) != u32::from_le_bytes(read(bytes, end, "checksum")?) {
        return Err(malformed(
            "checkpoint CRC-32C mismatch",
            scioxide_core::argument!("bytes"),
        ));
    }
    if header[9] & !allowed_flags != 0 || header[10..12] != [0, 0] {
        return Err(malformed(
            "noncanonical checkpoint flags or reserved bytes",
            scioxide_core::argument!("bytes"),
        ));
    }
    Ok(header[9])
}

/// Check the exact envelope size before writing, then encode an infallible
/// payload and the shared CRC trailer. The callback receives only the payload.
pub(crate) fn write_envelope(
    out: &mut [u8],
    algorithm: u8,
    version: u8,
    payload_bytes: usize,
    flags: u8,
    write_payload: impl FnOnce(&mut [u8]),
) -> Result<(), SciOxideError> {
    let payload = u32::try_from(payload_bytes).map_err(|_| {
        malformed(
            "checkpoint payload exceeds u32",
            scioxide_core::argument!("self"),
        )
    })?;
    let len = payload_bytes
        .checked_add(PREFIX_BYTES + TRAILER_BYTES)
        .ok_or_else(|| malformed("checkpoint size overflow", scioxide_core::argument!("self")))?;
    if out.len() != len {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(len),
            scioxide_core::Dimensions::one(out.len()),
        ));
    }
    out[..4].copy_from_slice(&MAGIC);
    out[4..6].copy_from_slice(&1_u16.to_le_bytes());
    out[6] = algorithm;
    out[7] = version;
    out[8] = 1;
    out[9] = flags;
    out[10..12].fill(0);
    out[12..16].copy_from_slice(&payload.to_le_bytes());
    let end = len - TRAILER_BYTES;
    write_payload(&mut out[PREFIX_BYTES..end]);
    let checksum = crc32c(&out[..end]);
    out[end..].copy_from_slice(&checksum.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checksum_matches_independent_castagnoli_check_values() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(&[0; 32]), 0x8a91_36aa);
        assert_eq!(crc32c(&[0xff; 32]), 0x62a8_ab43);
    }
}
