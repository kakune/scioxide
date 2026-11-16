//! Counter-based generators Philox4x64-10 and Threefry4x64-20 (Salmon et al.
//! 2011) with tree-structured streams.

use scioxide_core::{InvalidInputReason, RealScalar, SciOxideError};

use crate::state::{
    ENVELOPE_BYTES, PREFIX_BYTES, malformed, read, validate_envelope, write_envelope,
};
use crate::{
    BitGenerator, GeneratorAlgorithm, JumpableRng, SeedableRng, SerializableState, SkipAhead,
    SplittableRng, StreamIdentity,
};

// Each node owns 2^64 four-word blocks (the block index occupies counter word
// 0), hence 2^66 native words. The limit is inclusive in `seek` so the
// exhausted end position itself is representable and serializable.
const WORD_LIMIT: u128 = 1_u128 << 66;

/// Computes the Philox4x64-10 block function at a counter and key.
///
/// Philox4x64 maps a 256-bit counter and a 128-bit key to four 64-bit words with ten rounds. Each
/// round takes two 64 × 64 → 128-bit products of counter words with fixed multipliers and XORs
/// their high halves with the other counter words and the round key. The key is incremented by
/// the Weyl constants `0x9e3779b97f4a7c15` and `0xbb67ae8584caa73b` between rounds. Word order,
/// constants and outputs follow Salmon et al. (2011). Every counter and key is valid. The cost is
/// ten rounds, independent of the inputs.
///
/// # References
///
/// J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: as easy as
/// 1, 2, 3", *Proc. SC11*, ACM, 2011.
///
/// # Examples
///
/// ```
/// use scioxide_random::philox4x64_block;
/// // Known-answer vector of Salmon et al. (2011) for the zero counter and key.
/// let block = philox4x64_block([0; 4], [0; 2]);
/// assert_eq!(block, [
///     0x1655_4d9e_ca36_314c, 0xdb20_fe9d_672d_0fdc,
///     0xd7e7_72ce_e186_176b, 0x7e68_b68a_ec7b_a23b,
/// ]);
/// ```
#[must_use]
#[inline]
pub fn philox4x64_block(counter: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    philox4x64_rounds(counter, &philox4x64_schedule(key))
}

/// Philox4x64-10 round keys. The key is incremented by the Weyl constants
/// (golden ratio, sqrt(3) − 1) between rounds only, not after the last round.
const fn philox4x64_schedule(key: [u64; 2]) -> [[u64; 2]; 10] {
    let mut schedule = [key; 10];
    let mut round = 1;
    while round < 10 {
        let [first, second] = schedule[round - 1];
        schedule[round] = [
            first.wrapping_add(0x9e37_79b9_7f4a_7c15),
            second.wrapping_add(0xbb67_ae85_84ca_a73b),
        ];
        round += 1;
    }
    schedule
}

#[inline(always)]
fn philox4x64_rounds(mut counter: [u64; 4], schedule: &[[u64; 2]; 10]) -> [u64; 4] {
    // Each round is two 64x64->128 multiplies whose high halves are XOR-ed
    // with the other counter words and the round key; the output permutation
    // is Random123's word order.
    for key in schedule {
        let first = u128::from(counter[0]) * 0xd2e7_470e_e14c_6c93_u128;
        let second = u128::from(counter[2]) * 0xca5a_8263_9512_1157_u128;
        counter = [
            (second >> 64) as u64 ^ counter[1] ^ key[0],
            second as u64,
            (first >> 64) as u64 ^ counter[3] ^ key[1],
            first as u64,
        ];
    }
    counter
}

/// Computes the Threefry4x64-20 block function at a counter and key.
///
/// Threefry4x64 maps a 256-bit counter and a 256-bit key to four 64-bit words with twenty rounds
/// of the add-rotate-XOR mix of the Threefish block cipher. The Threefish key schedule is added
/// every four rounds. Word order, rotation constants and outputs follow Salmon et al. (2011).
/// Every counter and key is valid. The cost is twenty rounds, independent of the inputs.
///
/// # References
///
/// J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: as easy as
/// 1, 2, 3", *Proc. SC11*, ACM, 2011.
///
/// # Examples
///
/// ```
/// use scioxide_random::threefry4x64_block;
/// // Known-answer vector of Salmon et al. (2011) for the zero counter and key.
/// let block = threefry4x64_block([0; 4], [0; 4]);
/// assert_eq!(block, [
///     0x0921_8ebd_e6c8_5537, 0x5594_1f52_66d8_6105,
///     0x4bd2_5e16_2824_34dc, 0xee29_ec84_6bd2_e40b,
/// ]);
/// ```
#[must_use]
#[inline]
pub fn threefry4x64_block(counter: [u64; 4], key: [u64; 4]) -> [u64; 4] {
    const ROTATIONS: [[u32; 2]; 8] = [
        [14, 16],
        [52, 57],
        [23, 40],
        [5, 37],
        [25, 33],
        [46, 12],
        [58, 22],
        [32, 32],
    ];
    // Threefish key schedule: a fifth key word is the XOR of the key with the
    // constant C240, and every four rounds a rotated window of the extended
    // key plus the injection count is added. The initial addition below is
    // injection 0.
    let parity = key.iter().fold(0x1bd1_1bda_a9fc_1a22, |p, &word| p ^ word);
    let schedule = [key[0], key[1], key[2], key[3], parity];
    let mut words = core::array::from_fn(|i| counter[i].wrapping_add(key[i]));
    for round in 0..20 {
        let [a, b] = if round % 2 == 0 { [1, 3] } else { [3, 1] };
        let rotations = ROTATIONS[round % 8];
        words[0] = words[0].wrapping_add(words[a]);
        words[a] = words[a].rotate_left(rotations[0]) ^ words[0];
        words[2] = words[2].wrapping_add(words[b]);
        words[b] = words[b].rotate_left(rotations[1]) ^ words[2];
        if round % 4 == 3 {
            let injection = (round + 1) / 4;
            words.iter_mut().enumerate().for_each(|(i, word)| {
                *word = word.wrapping_add(schedule[(injection + i) % 5]);
            });
            words[3] = words[3].wrapping_add(injection as u64);
        }
    }
    words
}

// Threefry injects its key schedule inside the rounds; the key is the schedule.
const fn threefry4x64_schedule(key: [u64; 4]) -> [u64; 4] {
    key
}

#[inline]
fn threefry4x64_rounds(counter: [u64; 4], schedule: &[u64; 4]) -> [u64; 4] {
    threefry4x64_block(counter, *schedule)
}

/// Stream selectors and position of a counter-based generator.
#[derive(Debug, Clone)]
struct Cursor {
    stream: u64,
    subsequence: u64,
    path: u64,
    position: u128,
    block: [u64; 4],
}

impl Cursor {
    const fn root() -> Self {
        Self {
            stream: 0,
            subsequence: 0,
            path: 1,
            position: 0,
            block: [0; 4],
        }
    }

    // Counter-word layout gives domain separation: the block index varies
    // fastest, while path, stream and subsequence select disjoint counter
    // spaces, so distinct nodes can never evaluate the same block.
    fn counter(&self) -> Option<[u64; 4]> {
        (self.position < WORD_LIMIT).then_some([
            (self.position / 4) as u64,
            self.path,
            self.stream,
            self.subsequence,
        ])
    }

    /// Returns the end position after `count` more words, or an error past the limit.
    fn target(&self, count: u128) -> Result<u128, SciOxideError> {
        self.position
            .checked_add(count)
            .filter(|&p| p <= WORD_LIMIT)
            .ok_or_else(exhausted)
    }

    /// Child path code; a full parent path is invalid input.
    fn fork_path(&self, child: u8) -> Result<u64, SciOxideError> {
        // The path is a leading 1 sentinel followed by byte labels, which makes
        // the path encoding unambiguous (`[0]` differs from `[]`). A sentinel at
        // bit 56 already carries seven labels; one more would overflow u64.
        if self.path >= 1_u64 << 56 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::OutOfRange,
                "counter stream trees support at most seven child labels",
            ));
        }
        Ok((self.path << 8) | u64::from(child))
    }
}

fn valid_path(mut path: u64) -> bool {
    while path > 1 {
        path >>= 8;
    }
    path == 1
}

#[cold]
#[inline(never)]
fn exhausted() -> SciOxideError {
    SciOxideError::out_of_bounds(
        scioxide_core::quantity!("counter_stream_exhausted"),
        usize::MAX,
        usize::MAX,
    )
}

// Share protocol delegation and cursor commits, never algorithm dispatch.
macro_rules! counter_engine {
    (
        $(#[$attr:meta])*
        $name:ident,
        $key_words:expr,
        $schedule:ty,
        $prepare:ident,
        $rounds:ident
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        #[must_use]
        pub struct $name {
            key: [u64; $key_words],
            /// Round keys prepared from `key`.
            schedule: $schedule,
            cursor: Cursor,
        }

        impl $name {
            /// Number of 64-bit words in one stream, `2⁶⁶` (`2⁶⁴` blocks of four words).
            pub const WORD_CAPACITY: u128 = WORD_LIMIT;

            /// Creates the root stream at word zero from a key.
            pub const fn from_key(key: [u64; $key_words]) -> Self {
                Self {
                    key,
                    schedule: $prepare(key),
                    cursor: Cursor::root(),
                }
            }

            /// Returns the key.
            pub const fn key(&self) -> &[u64; $key_words] {
                &self.key
            }

            /// Returns the tree path as a code: a leading `0x01` byte followed by zero to seven
            /// child labels, one byte each. The root stream has code 1.
            pub const fn path_code(&self) -> u64 {
                self.cursor.path
            }

            /// Returns the counter of the block that holds the next word, or `None` if the stream
            /// is exhausted.
            ///
            /// The counter is `[block, path code, stream, subsequence]`, where `block` is the word
            /// position divided by four.
            pub fn counter(&self) -> Option<[u64; 4]> {
                self.cursor.counter()
            }

            /// Returns child `child` of this stream's tree path, at word zero.
            ///
            /// The parent is not advanced. Distinct paths use disjoint counter inputs;
            /// individual 64-bit output values can still repeat.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if this path already has seven labels.
            pub fn fork(&self, child: u8) -> Result<Self, SciOxideError> {
                let mut next = self.clone();
                next.cursor.path = self.cursor.fork_path(child)?;
                next.seek(0)?;
                Ok(next)
            }

            /// Returns the stream at tree path `path`, at word zero, with this stream's key, stream
            /// and subsequence.
            ///
            /// The path is absolute: the current path and position are ignored.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `path` has more than seven labels.
            pub fn derive(&self, path: &[u8]) -> Result<Self, SciOxideError> {
                if path.len() > 7 {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("path"),
                        InvalidInputReason::OutOfRange,
                        "counter stream trees support at most seven child labels",
                    ));
                }
                let mut next = self.clone();
                next.cursor.path = path
                    .iter()
                    .fold(1, |code, &child| (code << 8) | u64::from(child));
                next.seek(0)?;
                Ok(next)
            }

            /// Moves to the absolute word position `position`, forward or backward, in O(1).
            ///
            /// `position` may equal `WORD_CAPACITY`, the exhausted end of the stream. Unlike
            /// [`SkipAhead::skip_to`](crate::SkipAhead::skip_to), this can move backward to
            /// replay words.
            ///
            /// # Errors
            ///
            /// - `OutOfBounds` if `position` exceeds `WORD_CAPACITY`. The generator is unchanged.
            pub fn seek(&mut self, position: u128) -> Result<(), SciOxideError> {
                if position > WORD_LIMIT {
                    return Err(exhausted());
                }
                self.cursor.position = position;
                // `draw_admitted` only evaluates a block at lane 0, so a seek
                // into the middle of a block must materialize it here. At a
                // block boundary the cached block is stale by design and will be
                // regenerated on the next draw.
                self.cursor.block = if position % 4 != 0 && position < WORD_LIMIT {
                    $rounds(
                        [
                            (position / 4) as u64,
                            self.cursor.path,
                            self.cursor.stream,
                            self.cursor.subsequence,
                        ],
                        &self.schedule,
                    )
                } else {
                    [0; 4]
                };
                Ok(())
            }

            fn identity(&self) -> StreamIdentity<'static> {
                let mut origin = [0; 6];
                origin[..$key_words].copy_from_slice(&self.key);
                origin[4] = self.cursor.path;
                StreamIdentity::new(
                    GeneratorAlgorithm::$name,
                    origin,
                    Some((self.cursor.stream, self.cursor.subsequence)),
                )
            }

            // Called only after complete capacity admission. Integer kernels are pure;
            // accepted cursor advancement is explicit here.
            #[inline]
            fn draw_admitted(&mut self) -> u64 {
                #[allow(clippy::cast_possible_truncation)]
                let lane = self.cursor.position as usize & 3;
                if lane == 0 {
                    self.cursor.block = $rounds(
                        [
                            (self.cursor.position / 4) as u64,
                            self.cursor.path,
                            self.cursor.stream,
                            self.cursor.subsequence,
                        ],
                        &self.schedule,
                    );
                }
                let word = self.cursor.block[lane];
                self.cursor.position += 1;
                word
            }

            /// Fills floats from any position, after the attempt has been counted.
            #[inline(never)]
            fn fill_float_slice<S: RealScalar>(
                &mut self,
                output: &mut [S],
                attempts: &mut usize,
            ) -> Result<(), SciOxideError> {
                let requested = output.len();
                let available = usize::try_from(WORD_LIMIT - self.cursor.position)
                    .unwrap_or(usize::MAX).min(requested);
                // Leading/trailing partial blocks retain scalar cache behavior.
                *attempts += available;
                #[allow(clippy::cast_possible_truncation)]
                let lane = self.cursor.position as usize & 3;
                let leading = ((4 - lane) & 3).min(available);
                let (prefix, values) = output[..available].split_at_mut(leading);
                for value in prefix {
                    *value = crate::decode_u64(self.draw_admitted());
                }
                let mut blocks = values.chunks_exact_mut(4);
                for block in blocks.by_ref() {
                    self.cursor.block = $rounds(
                        [(self.cursor.position / 4) as u64, self.cursor.path,
                         self.cursor.stream, self.cursor.subsequence],
                        &self.schedule,
                    );
                    self.cursor.position += 4;
                    for (value, word) in block.iter_mut().zip(self.cursor.block) {
                        *value = crate::decode_u64(word);
                    }
                }
                for value in blocks.into_remainder() {
                    *value = crate::decode_u64(self.draw_admitted());
                }
                if available < requested {
                    *attempts += 1;
                    return Err(exhausted());
                }
                Ok(())
            }
        }

        impl BitGenerator for $name {
            #[inline]
            fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
                self.next_u32()
            }
            #[inline]
            fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
                self.next_u64()
            }
            #[inline]
            fn position(&self) -> Option<u128> {
                Some(self.cursor.position)
            }
            fn stream_identity(&self) -> Option<StreamIdentity<'_>> {
                Some(self.identity())
            }
            #[inline]
            fn next_u64(&mut self) -> Result<u64, SciOxideError> {
                // One word fits exactly when the cursor is below the inclusive end.
                if self.cursor.position >= WORD_LIMIT {
                    return Err(exhausted());
                }
                Ok(self.draw_admitted())
            }
            #[inline]
            fn next_u32(&mut self) -> Result<u32, SciOxideError> {
                Ok((self.next_u64()? >> 32) as u32)
            }
            fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
                // Admit the whole request before publishing or advancing.
                self.cursor.target(output.len().div_ceil(8) as u128)?;
                crate::util::fill_bytes_with(output, 8, || Ok(self.draw_admitted()))
            }
        }

        impl<S: RealScalar> crate::float::FloatSource<S> for $name {
            #[inline]
            fn draw_float(&mut self) -> Result<S, SciOxideError> {
                self.next_u64().map(crate::decode_u64)
            }

            #[inline]
            fn fill_floats(&mut self, output: &mut [S], attempts: &mut usize) -> Result<(), SciOxideError> {
                if output.len() == 4 && self.cursor.position & 3 == 0
                    && self.cursor.position <= WORD_LIMIT - 4
                {
                    // One complete counter block needs no general prefix/tail
                    // machinery or loop-invariant schedule hoisting.
                    self.cursor.block = $rounds(
                        [(self.cursor.position / 4) as u64, self.cursor.path,
                         self.cursor.stream, self.cursor.subsequence],
                        &self.schedule,
                    );
                    self.cursor.position += 4;
                    *attempts += 4;
                    for (value, word) in output.iter_mut().zip(self.cursor.block) {
                        *value = crate::decode_u64(word);
                    }
                    Ok(())
                } else {
                    self.fill_float_slice(output, attempts)
                }
            }
        }

        impl SeedableRng for $name {
            fn from_seed(seed: u64) -> Self {
                let mut mixer = super::SplitMix64::from_seed(seed);
                Self::from_key(core::array::from_fn(|_| mixer.next_word()))
            }
        }

        impl SkipAhead for $name {
            fn skip_ahead(&mut self, count: u128) -> Result<(), SciOxideError> {
                self.seek(self.cursor.target(count)?)
            }
            // Every word or float read is one native transition.
            #[inline]
            fn read_transition_bound(&self, _: crate::traits::private::Token) -> Option<u128> {
                Some(1)
            }
        }

        impl JumpableRng for $name {
            fn jump(&mut self) -> Result<(), SciOxideError> {
                self.skip_ahead(1_u128 << 32)
            }
            fn long_jump(&mut self) -> Result<(), SciOxideError> {
                self.skip_ahead(1_u128 << 48)
            }
        }

        impl SplittableRng for $name {
            fn split(&self, stream: u64) -> Result<Self, SciOxideError> {
                self.substream(stream, 0)
            }
            fn substream(&self, stream: u64, subsequence: u64) -> Result<Self, SciOxideError> {
                // Selectors are counter words, so a substream is just a fresh root
                // cursor under the same key; no state needs to be advanced.
                let mut next = Self::from_key(self.key);
                next.cursor.stream = stream;
                next.cursor.subsequence = subsequence;
                Ok(next)
            }
        }

        impl SerializableState for $name {
            const STATE_BYTES: usize = ENVELOPE_BYTES + 8 * ($key_words + 3) + 16;
            fn export_state_into(&self, output: &mut [u8]) -> Result<(), SciOxideError> {
                write_envelope(
                    output,
                    GeneratorAlgorithm::$name as u8,
                    1,
                    Self::STATE_BYTES - ENVELOPE_BYTES,
                    0,
                    |output| {
                        let mut offset = 0;
                        for word in self.key.iter().chain([
                            &self.cursor.stream,
                            &self.cursor.subsequence,
                            &self.cursor.path,
                        ]) {
                            output[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
                            offset += 8;
                        }
                        output[offset..offset + 16]
                            .copy_from_slice(&self.cursor.position.to_le_bytes());
                    },
                )
            }
            fn import_state(bytes: &[u8]) -> Result<Self, SciOxideError> {
                validate_envelope(
                    bytes,
                    GeneratorAlgorithm::$name as u8,
                    1..=1,
                    Some(Self::STATE_BYTES - ENVELOPE_BYTES),
                    0,
                )?;
                let mut offset = PREFIX_BYTES;
                let mut key = [0; $key_words];
                for word in &mut key {
                    *word = u64::from_le_bytes(read(bytes, offset, "key")?);
                    offset += 8;
                }
                let stream = u64::from_le_bytes(read(bytes, offset, "stream")?);
                let subsequence = u64::from_le_bytes(read(bytes, offset + 8, "subsequence")?);
                let path = u64::from_le_bytes(read(bytes, offset + 16, "path")?);
                let position = u128::from_le_bytes(read(bytes, offset + 24, "position")?);
                if !valid_path(path) || position > WORD_LIMIT {
                    return Err(malformed(
                        "invalid counter path or position",
                        scioxide_core::argument!("bytes"),
                    ));
                }
                let mut next = Self::from_key(key);
                next.cursor.stream = stream;
                next.cursor.subsequence = subsequence;
                next.cursor.path = path;
                next.seek(position)?;
                Ok(next)
            }
        }

        crate::transaction::clone_snapshot!($name);
    };
}

counter_engine!(
    /// Counter-based generator Philox4x64-10 with 64-bit output and 2⁶⁶ words per stream.
    ///
    /// Word `n` of a stream is lane `n mod 4` of [`philox4x64_block`](crate::philox4x64_block) at
    /// the counter `[⌊n/4⌋, path, stream, subsequence]` and the key, with ten rounds (Salmon et
    /// al. 2011). Every word is a function of its counter alone, so
    /// [`SkipAhead`](crate::SkipAhead), [`seek`](Self::seek),
    /// [`split`](crate::SplittableRng::split), [`fork`](Self::fork) and [`derive`](Self::derive)
    /// all cost O(1). Under one key, streams with different stream, subsequence or path selectors
    /// use disjoint counter inputs. Individual output words can still repeat.
    /// [`jump`](crate::JumpableRng::jump) advances 2³² words and
    /// [`long_jump`](crate::JumpableRng::long_jump) 2⁴⁸ words.
    /// [`next_u32`](crate::BitGenerator::next_u32) returns the high 32 bits of a word.
    /// [`from_seed`](crate::SeedableRng::from_seed) takes the key from
    /// [`SplitMix64`](crate::SplitMix64) outputs, and [`from_key`](Self::from_key) takes it
    /// directly.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` when a draw would pass the end of the stream (`WORD_CAPACITY` words).
    ///   [`fill_bytes`](crate::BitGenerator::fill_bytes) checks the whole request first.
    ///
    /// # References
    ///
    /// J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: as easy as
    /// 1, 2, 3", *Proc. SC11*, ACM, 2011.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{
    ///     BitGenerator, JumpableRng, Philox4x64, SerializableState, SplittableRng,
    /// };
    /// use scioxide_random::philox4x64_block;
    /// let mut root = Philox4x64::from_key([1, 2]);
    /// assert_eq!(root.counter(), Some([0, 1, 0, 0])); // block, path, stream, subsequence
    /// assert_eq!(root.next_u64()?, philox4x64_block([0, 1, 0, 0], [1, 2])[0]);
    /// let mut worker = root.split(4)?.fork(3)?;
    /// worker.jump()?; // 2^32 words
    /// let checkpoint = worker.export_state()?;
    /// let word = worker.next_u64()?;
    /// worker.seek(1 << 32)?; // backward replay
    /// assert_eq!(worker.next_u64()?, word);
    /// assert_eq!(Philox4x64::import_state(&checkpoint)?.next_u64()?, word);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Philox4x64,
    2,
    [[u64; 2]; 10],
    philox4x64_schedule,
    philox4x64_rounds
);
counter_engine!(
    /// Counter-based generator Threefry4x64-20 with 64-bit output and 2⁶⁶ words per stream.
    ///
    /// Word `n` of a stream is lane `n mod 4` of [`threefry4x64_block`](crate::threefry4x64_block)
    /// at the counter `[⌊n/4⌋, path, stream, subsequence]` and the key, with twenty rounds
    /// (Salmon et al. 2011). Streams, paths, skipping and jumps work as for
    /// [`Philox4x64`](crate::Philox4x64), and all cost O(1).
    /// [`next_u32`](crate::BitGenerator::next_u32) returns the high 32 bits of a word.
    /// [`from_seed`](crate::SeedableRng::from_seed) takes the key from
    /// [`SplitMix64`](crate::SplitMix64) outputs, and [`from_key`](Self::from_key) takes it
    /// directly.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` when a draw would pass the end of the stream (`WORD_CAPACITY` words).
    ///
    /// # References
    ///
    /// J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: as easy as
    /// 1, 2, 3", *Proc. SC11*, ACM, 2011.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, SeedableRng, SerializableState, SkipAhead, Threefry4x64};
    /// use scioxide_random::threefry4x64_block;
    /// let seeded = Threefry4x64::from_seed(42);
    /// let key = *seeded.key();
    /// let mut rng = seeded.derive(&[2, 5])?;
    /// assert_eq!(rng.path_code(), 0x01_02_05);
    /// rng.skip_ahead(6)?; // word 6 is lane 2 of block 1
    /// let checkpoint = rng.export_state()?;
    /// let word = rng.next_u64()?;
    /// assert_eq!(word, threefry4x64_block([1, 0x01_02_05, 0, 0], key)[2]);
    /// assert_eq!(Threefry4x64::import_state(&checkpoint)?.next_u64()?, word);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Threefry4x64,
    4,
    [u64; 4],
    threefry4x64_schedule,
    threefry4x64_rounds
);
