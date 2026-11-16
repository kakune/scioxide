//! Canonical array seeding, luxury selection, borrowed floats and portable replay.
use scioxide_core::SciOxideError;
use scioxide_random::{BitGenerator, Mt19937, Ranlxd, RanlxdLuxury, Rng, SerializableState};
fn main() -> Result<(), SciOxideError> {
    let mut mt = Mt19937::from_seed_words(&[0x123, 0x234, 0x345, 0x456])?;
    println!("canonical MT19937 word: {}", mt.next_u32()?);
    let mut luxury = Ranlxd::with_luxury(42, RanlxdLuxury::Level2);
    let checkpoint = luxury.export_state()?;
    let single: f32 = luxury.next_float()?;
    let double: f64 = luxury.next_float()?;
    println!("RANLXD real samples: {single}, {double}");
    luxury.restore_state(&checkpoint)?;
    let mut replay = Ranlxd::import_state(&checkpoint)?;
    assert_eq!(luxury.next_u64()?, replay.next_u64()?);
    assert_eq!(luxury.stream_identity(), replay.stream_identity());
    Ok(())
}
