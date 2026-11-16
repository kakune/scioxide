//! Explicit counter stream trees and portable continuation, without ambient identity.
use scioxide_core::SciOxideError;
use scioxide_random::{
    BitGenerator, BoundedSubstream, Distribution, Philox4x64, SeedableRng, SerializableState,
    SplittableRng, Threefry4x64, Uniform,
};

fn main() -> Result<(), SciOxideError> {
    let root = Philox4x64::from_seed(42).substream(12, 3)?;
    let mut child = root.derive(&[2, 7])?;
    let checkpoint = child.export_state()?;
    let first = child.next_u64()?;
    child.restore_state(&checkpoint)?;
    assert_eq!(first, child.next_u64()?);
    println!("Philox path {} starts with {first:016x}", child.path_code());

    // Partition by native words. Bounded streams can feed existing distributions.
    let root = Threefry4x64::from_key([1, 2, 3, 4]);
    let law = Uniform::new(0.0_f64, 1.0)?;
    let mut chunk = BoundedSubstream::for_chunk(&root, 5, 4)?;
    for _ in 0..4 {
        println!("{}", law.sample(&mut chunk)?);
    }
    assert!(
        (chunk.next_u64())
            .is_err_and(|error| error.kind() == scioxide_core::SciOxideErrorKind::OutOfBounds)
    );
    Ok(())
}
