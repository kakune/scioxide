//! Prepared engine and checkpoint operations require no heap allocation.
#![allow(clippy::unwrap_used)]

use scioxide_random::{
    BitGenerator, Lcg64, Mrg32k3a, Pcg64, Rng, SeedableRng, SerializableState, SplitMix64,
    Xoshiro128StarStar, Xoshiro256StarStar,
};

fn check<B: SeedableRng + SerializableState + Rng<f32> + Rng<f64>>() {
    let mut rng = B::from_seed(42);
    let snapshot = rng.export_state().unwrap();
    let mut output = vec![0; B::STATE_BYTES];
    {
        let _: f32 = rng.next_float().unwrap();
        let _: f64 = rng.next_float().unwrap();
        rng.next_u32().unwrap();
        rng.next_u64().unwrap();
        rng.fill_bytes(&mut [0; 17]).unwrap();
    };
    rng.export_state_into(&mut output).unwrap();
    B::import_state(&output).unwrap();
    rng.restore_state(&snapshot).unwrap();
    assert_eq!(rng.export_state().unwrap(), snapshot);
}
#[test]
fn all_builtin_prepared_operations_execute() {
    check::<SplitMix64>();
    check::<Xoshiro256StarStar>();
    check::<Xoshiro128StarStar>();
    check::<Pcg64>();
    check::<Lcg64>();
    check::<Mrg32k3a>();
    check::<scioxide_random::Philox4x64>();
    check::<scioxide_random::Threefry4x64>();
}

#[test]
fn counter_tree_movement_allocates_nothing() {
    use scioxide_random::{JumpableRng, SkipAhead, SplittableRng};
    {
        let mut philox = scioxide_random::Philox4x64::from_seed(42)
            .substream(1, 2)
            .unwrap()
            .derive(&[0, 255])
            .unwrap()
            .fork(9)
            .unwrap();
        philox.seek(3).unwrap();
        philox.jump().unwrap();
        philox.long_jump().unwrap();
        philox.skip_ahead(17).unwrap();
        let mut threefry = scioxide_random::Threefry4x64::from_seed(42)
            .substream(1, 2)
            .unwrap()
            .derive(&[0, 255])
            .unwrap()
            .fork(9)
            .unwrap();
        threefry.seek(3).unwrap();
        threefry.jump().unwrap();
        threefry.long_jump().unwrap();
        threefry.skip_ahead(17).unwrap();
        (philox.next_u64().unwrap(), threefry.next_u64().unwrap())
    };
}

#[test]
fn classic_prepared_operations_execute() {
    check::<scioxide_random::Mt19937>();
    check::<scioxide_random::Ranlux>();
    check::<scioxide_random::Ranlxs>();
    check::<scioxide_random::Ranlxd>();
    check::<scioxide_random::Taus2>();
    check::<scioxide_random::Taus113>();
    check::<scioxide_random::Mrg>();
    check::<scioxide_random::Cmrg>();
}
#[test]
fn gfsr_prepared_operations_execute() {
    check::<scioxide_random::Gfsr4>();
}
