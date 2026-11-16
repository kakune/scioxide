//! Shared storage and portable protocol for the classic engine families.
//!
//! Recurrences are implemented in the sibling family modules. Words include
//! fixed-width cursor/parameter fields; no Rust layout is part of the protocol.

use crate::state::NativeOrigin;
use crate::state::{
    ENVELOPE_BYTES, GeneratorAlgorithm, PREFIX_BYTES, StreamIdentity, malformed, read,
    validate_envelope, write_envelope,
};
use crate::util::TransitionCounter;

// Native array words: GFSR's large ring must not be widened in memory or on wire.
pub(super) trait StateWord: Copy + Default + Into<u64> + TryFrom<u64> {
    const BYTES: usize;
    fn identity(words: &[Self]) -> NativeOrigin<'_>;
}
impl StateWord for u64 {
    const BYTES: usize = 8;
    fn identity(words: &[Self]) -> NativeOrigin<'_> {
        NativeOrigin::Wide(words)
    }
}
impl StateWord for u32 {
    const BYTES: usize = 4;
    fn identity(words: &[Self]) -> NativeOrigin<'_> {
        NativeOrigin::Narrow(words)
    }
}
use scioxide_core::SciOxideError;

#[derive(Debug, Clone)]
pub(super) struct ClassicState<const N: usize, W = u64> {
    pub(super) words: [W; N],
    pub(super) origin: [W; N],
    pub(super) counter: TransitionCounter,
}
impl<const N: usize, W: StateWord> ClassicState<N, W> {
    pub(super) fn new(words: [W; N]) -> Self {
        Self {
            words,
            origin: words,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
    pub(super) fn identity(&self, algorithm: GeneratorAlgorithm) -> StreamIdentity<'_> {
        StreamIdentity {
            algorithm,
            origin: [0; 6],
            selectors: None,
            extended_origin: W::identity(&self.origin),
        }
    }
    pub(super) const PAYLOAD_BYTES: usize = 16 + 2 * W::BYTES * N;
    pub(super) const BYTES: usize = ENVELOPE_BYTES + Self::PAYLOAD_BYTES;
    pub(super) fn export(
        &self,
        algorithm: GeneratorAlgorithm,
        out: &mut [u8],
    ) -> Result<(), SciOxideError> {
        write_envelope(
            out,
            algorithm as u8,
            1,
            Self::PAYLOAD_BYTES,
            u8::from(self.counter.position().is_some()),
            |out| {
                for (chunk, word) in out[..2 * W::BYTES * N]
                    .chunks_exact_mut(W::BYTES)
                    .zip(self.origin.iter().chain(self.words.iter()))
                {
                    chunk.copy_from_slice(&(*word).into().to_le_bytes()[..W::BYTES]);
                }
                out[2 * W::BYTES * N..]
                    .copy_from_slice(&self.counter.position().unwrap_or(0).to_le_bytes());
            },
        )
    }

    pub(super) fn import(
        bytes: &[u8],
        algorithm: GeneratorAlgorithm,
        validate: impl Fn(&[W; N]) -> bool,
    ) -> Result<Self, SciOxideError> {
        let flags = validate_envelope(bytes, algorithm as u8, 1..=1, Some(Self::PAYLOAD_BYTES), 1)?;
        let mut origin = [W::default(); N];
        let mut words = [W::default(); N];
        for (i, word) in origin.iter_mut().chain(words.iter_mut()).enumerate() {
            let value = if W::BYTES == 4 {
                u64::from(u32::from_le_bytes(read(
                    bytes,
                    PREFIX_BYTES + 4 * i,
                    "state word",
                )?))
            } else {
                u64::from_le_bytes(read(bytes, PREFIX_BYTES + 8 * i, "state word")?)
            };
            *word = W::try_from(value).map_err(|_| {
                malformed(
                    "word exceeds native width",
                    scioxide_core::argument!("bytes"),
                )
            })?;
        }
        let count = u128::from_le_bytes(read(bytes, PREFIX_BYTES + 2 * W::BYTES * N, "counter")?);
        // Flag 0 means "position unknown", which is exported with a zero
        // counter; a nonzero counter under that flag is a corrupted record.
        if !validate(&origin) || !validate(&words) || (flags == 0 && count != 0) {
            return Err(malformed(
                "invalid recurrence, root, or counter",
                scioxide_core::argument!("bytes"),
            ));
        }
        Ok(Self {
            origin,
            words,
            counter: TransitionCounter::from_position((flags == 1).then_some(count)),
        })
    }
}

// Only protocol delegation is shared; numerical kernels remain family-specific.
macro_rules! classic_engine {
    ($(#[$attr:meta])* $name:ident, $n:expr, $bits:expr) => {
        classic_engine!($(#[$attr])* $name, $n, $bits, u64);
    };
    ($(#[$attr:meta])* $name:ident, $n:expr, $bits:expr, $word:ty) => {
        classic_engine!($(#[$attr])* $name, $n, $bits, $word, 1_u64 << $bits);
    };
    ($(#[$attr:meta])* $name:ident, $n:expr, $bits:expr, $word:ty, $radix:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        #[must_use]
        pub struct $name(pub(super) super::classic::ClassicState<$n, $word>);
        impl crate::BitGenerator for $name {
            #[inline]
            fn position(&self) -> Option<u128> {
                self.0.counter.position()
            }
            fn stream_identity(&self) -> Option<crate::StreamIdentity<'_>> {
                Some(self.0.identity(crate::GeneratorAlgorithm::$name))
            }
            #[inline]
            fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
                // Wider native outputs keep their top 32 bits, the best-mixed bits
                // of these recurrences; narrow ones are returned unchanged.
                #[allow(clippy::cast_possible_truncation)]
                let word = if $bits > 32 {
                    self.next_native() >> ($bits as u32).saturating_sub(32)
                } else {
                    self.next_native()
                } as u32;
                Ok(word)
            }
            #[inline]
            fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
                Ok(if $bits > 32 {
                    self.next_native()
                } else {
                    let low = self.next_native();
                    low | (self.next_native() << 32)
                })
            }
            #[inline]
            fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
                if $radix == 1_u64 << $bits && $bits >= 32 {
                    self.next_u32()
                } else {
                    #[allow(clippy::cast_possible_truncation)]
                    crate::prng::words::uniform_word(32, $radix, || Ok(self.next_native()))
                        .map(|word| word as u32)
                }
            }
            #[inline]
            fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
                if $bits == 32 && $radix == 1_u64 << 32 {
                    self.next_u64()
                } else {
                    crate::prng::words::uniform_word(64, $radix, || Ok(self.next_native()))
                }
            }
            fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
                crate::util::fill_bytes_with(out, ($bits as usize).div_ceil(8), || {
                    Ok(self.next_native())
                })
            }
        }
        impl crate::SerializableState for $name {
            const STATE_BYTES: usize = super::classic::ClassicState::<$n, $word>::BYTES;
            fn export_state_into(&self, out: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
                self.0.export(crate::GeneratorAlgorithm::$name, out)
            }
            fn import_state(bytes: &[u8]) -> Result<Self, scioxide_core::SciOxideError> {
                let state = super::classic::ClassicState::import(
                    bytes,
                    crate::GeneratorAlgorithm::$name,
                    Self::valid,
                )?;
                if !Self::consistent(&state.origin, &state.words) {
                    return Err(crate::state::malformed(
                        "root and current parameters differ",
                        scioxide_core::argument!("bytes"),
                    ));
                }
                Ok(Self(state))
            }
        }
        crate::transaction::clone_snapshot!($name);
    };
}
pub(super) use classic_engine;

macro_rules! dyadic_float {
    ($name:ident, $bits:expr) => {
        impl<S: scioxide_core::RealScalar> crate::float::FloatSource<S> for $name {
            #[inline]
            fn draw_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
                // Left-align the `$bits`-wide native output so the shared decoder
                // takes its high bits, giving a dyadic value in [0, 1).
                Ok(crate::decode_u64(self.next_native() << (64 - $bits)))
            }
        }
    };
}
pub(super) use dyadic_float;
