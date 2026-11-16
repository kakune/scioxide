//! Known-answer contracts for SciOxide's built-in random engines.
//!
//! All fixtures use the documented reference seed 42. Expected values are
//! literal constants so a transition, seeding, jump, scrambling, conversion,
//! or serialization change cannot regenerate its own passing fixture.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::hash::Hash;

use scioxide_random::{
    BitGenerator, JumpableRng, Lcg64, Mrg32k3a, Pcg64, Rng, SeedableRng, SplitMix64, SplittableRng,
    Xoshiro128StarStar, Xoshiro256StarStar,
};

const REFERENCE_SEED: u64 = 42;
const VECTOR_LENGTH: usize = 8;
const STREAM_PREFIX_LENGTH: usize = 64;

const SPLITMIX64: [u64; VECTOR_LENGTH] = [
    13_679_457_532_755_275_413,
    2_949_826_092_126_892_291,
    5_139_283_748_462_763_858,
    6_349_198_060_258_255_764,
    701_532_786_141_963_250,
    16_015_981_125_662_989_062,
    4_028_864_712_777_624_925,
    14_769_051_326_987_775_908,
];

const XOSHIRO256: [u64; VECTOR_LENGTH] = [
    1_546_998_764_402_558_742,
    6_990_951_692_964_543_102,
    12_544_586_762_248_559_009,
    17_057_574_109_182_124_193,
    18_295_552_978_065_317_476,
    14_199_186_830_065_750_584,
    13_267_978_908_934_200_754,
    15_679_888_225_317_814_407,
];

const XOSHIRO128: [u32; VECTOR_LENGTH] = [
    1_776_835_114,
    4_165_204_688,
    17_111_135,
    2_317_295_270,
    2_792_088_233,
    2_554_630_222,
    2_940_343_271,
    2_244_566_231,
];

const PCG64: [u64; VECTOR_LENGTH] = [
    2_915_081_201_720_324_186,
    13_533_757_442_135_995_717,
    13_172_715_927_431_628_928,
    13_789_878_565_430_171_748,
    8_308_839_764_963_933_125,
    11_940_931_540_510_907_511,
    7_885_000_673_997_492_397,
    9_950_184_437_018_364_971,
];

const LCG64: [u64; VECTOR_LENGTH] = [
    10_481_999_410_520_546_993,
    4_159_066_171_780_167_020,
    7_615_522_811_268_512_075,
    11_628_791_489_956_661_374,
    12_546_512_532_490_043_765,
    483_838_003_013_946_848,
    401_415_969_518_064_047,
    2_812_299_150_962_093_586,
];

const MRG32K3A: [u64; VECTOR_LENGTH] = [
    12_957_944_158_708_768_729,
    18_167_628_214_027_766_181,
    11_994_439_707_716_979_413,
    13_259_518_751_684_942_344,
    18_429_585_954_644_498_827,
    4_821_671_168_068_222_652,
    16_479_752_842_384_901_948,
    12_786_426_773_760_039_499,
];
const MRG32K3A_SEED_42_STATE: [u64; 6] = [
    748_412_454,
    507_480_424,
    1_297_493_171,
    3_814_460_716,
    1_072_842_093,
    4_115_200_557,
];

// Generated independently with upstream RngStreams commit
// b47f576ee28951bc437b3ed686ff4b81a96982c0 from its default six-word state.
const MRG32K3A_REFERENCE_WORDS: [u32; VECTOR_LENGTH] = [
    545_508_589,
    1_368_065_410,
    1_327_943_761,
    3_546_985_096,
    951_893_194,
    2_290_915_636,
    2_064_909_380,
    1_527_117_980,
];
const MRG32K3A_REFERENCE_F64: [u64; VECTOR_LENGTH] = [
    0x3fc0_41e6_83b5_8b4b,
    0x3fd4_62c1_7110_3d2c,
    0x3fd3_c9b4_2453_e25e,
    0x3fea_6d56_6678_d634,
    0x3fcc_5e5e_7c0c_acc6,
    0x3fe1_1193_345e_479b,
    0x3fde_c501_2a00_10f3,
    0x3fd6_c17e_397d_368f,
];
const MRG32K3A_REFERENCE_F32: [u32; VECTOR_LENGTH] = [
    1_040_322_352,
    1_050_875_402,
    1_050_561_952,
    1_062_431_410,
    1_046_672_112,
    1_057_524_889,
    1_056_319_496,
    1_052_118_000,
];
const MRG32K3A_REFERENCE_FILL_64: [u8; 64] = [
    237, 204, 131, 32, 130, 5, 139, 81, 81, 208, 38, 79, 136, 178, 106, 211, 202, 188, 188, 56, 52,
    153, 140, 136, 68, 4, 20, 123, 156, 248, 5, 91, 211, 34, 208, 34, 162, 135, 127, 193, 112, 151,
    87, 147, 138, 245, 249, 104, 65, 87, 138, 83, 89, 106, 137, 61, 239, 21, 45, 156, 214, 101,
    120, 231,
];
const MRG32K3A_STREAM_1_STATE: [u64; 6] = [
    3_692_455_944,
    1_366_884_236,
    2_968_912_127,
    335_948_734,
    4_161_675_175,
    475_798_818,
];
const MRG32K3A_SUBSTREAM_0_1_STATE: [u64; 6] = [
    870_504_860,
    2_641_697_727,
    884_013_853,
    339_352_413,
    2_374_306_706,
    3_651_603_887,
];
const MRG32K3A_SUBSTREAM_1_1_STATE: [u64; 6] = [
    3_119_395_571,
    2_178_405_402,
    1_065_030_501,
    3_980_307_777,
    2_117_495_919,
    1_836_828_492,
];

// First eight points in dimensions 1--8 from the canonical Joe--Kuo D(6)
// prefix (`new-joe-kuo-6.21201`, UNSW, 2010). These coordinate words are
// converted exactly to f64 by scaling with 2^-32.

fn next_u64_vector<R: BitGenerator>(rng: &mut R) -> [u64; VECTOR_LENGTH] {
    std::array::from_fn(|_| rng.next_u64().unwrap())
}

fn next_u32_vector<R: BitGenerator>(rng: &mut R) -> [u32; VECTOR_LENGTH] {
    std::array::from_fn(|_| rng.next_u32().unwrap())
}

fn mrg_state_words(rng: &Mrg32k3a) -> [u64; 6] {
    rng.state_words()
}

fn u64_prefix<R: BitGenerator>(mut rng: R) -> HashSet<u64> {
    (0..STREAM_PREFIX_LENGTH)
        .map(|_| rng.next_u64().unwrap())
        .collect()
}

fn u32_prefix<R: BitGenerator>(mut rng: R) -> HashSet<u32> {
    (0..STREAM_PREFIX_LENGTH)
        .map(|_| rng.next_u32().unwrap())
        .collect()
}

fn assert_pairwise_disjoint<T: Eq + Hash>(streams: &[HashSet<T>]) {
    assert!(
        streams
            .iter()
            .all(|stream| stream.len() == STREAM_PREFIX_LENGTH)
    );
    assert!(streams.iter().enumerate().all(|(index, stream)| {
        streams
            .iter()
            .skip(index + 1)
            .all(|other| stream.is_disjoint(other))
    }));
}

#[test]
fn all_prng_engines_match_seed_42_known_answers() {
    assert_eq!(
        next_u64_vector(&mut SplitMix64::from_seed(REFERENCE_SEED)),
        SPLITMIX64
    );
    assert_eq!(
        next_u64_vector(&mut Xoshiro256StarStar::from_seed(REFERENCE_SEED)),
        XOSHIRO256
    );
    assert_eq!(
        next_u32_vector(&mut Xoshiro128StarStar::from_seed(REFERENCE_SEED)),
        XOSHIRO128
    );
    assert_eq!(
        next_u64_vector(&mut Pcg64::from_seed(REFERENCE_SEED)),
        PCG64
    );
    assert_eq!(
        next_u64_vector(&mut Lcg64::from_seed(REFERENCE_SEED)),
        LCG64
    );
    let mut mrg32k3a = Mrg32k3a::from_seed(REFERENCE_SEED);
    assert_eq!(mrg_state_words(&mrg32k3a), MRG32K3A_SEED_42_STATE);
    assert_eq!(mrg32k3a.position(), Some(0));
    assert_eq!(next_u64_vector(&mut mrg32k3a), MRG32K3A);
}

#[test]
fn mrg32k3a_matches_upstream_words_floats_and_jump_boundaries() {
    let reference_state = [12_345; 6];
    let mut words = Mrg32k3a::from_state_words(reference_state).unwrap();
    assert_eq!(next_u32_vector(&mut words), MRG32K3A_REFERENCE_WORDS);

    let mut f64_generator = Mrg32k3a::from_state_words(reference_state).unwrap();
    let f64_bits = std::array::from_fn(|_| {
        <_ as Rng<f64>>::next_float(&mut &mut f64_generator)
            .unwrap()
            .to_bits()
    });
    assert_eq!(f64_bits, MRG32K3A_REFERENCE_F64);

    let mut f32_generator = Mrg32k3a::from_state_words(reference_state).unwrap();
    let f32_bits = std::array::from_fn(|_| {
        <_ as Rng<f32>>::next_float(&mut &mut f32_generator)
            .unwrap()
            .to_bits()
    });
    assert_eq!(f32_bits, MRG32K3A_REFERENCE_F32);

    let mut byte_generator = Mrg32k3a::from_state_words(reference_state).unwrap();
    let mut bytes = [0_u8; 64];
    byte_generator.fill_bytes(&mut bytes).unwrap();
    assert_eq!(bytes, MRG32K3A_REFERENCE_FILL_64);
    assert_eq!(byte_generator.position(), Some(16));

    let base = Mrg32k3a::from_state_words(reference_state).unwrap();
    let stream_1 = base.split(1).unwrap();
    assert_eq!(mrg_state_words(&stream_1), MRG32K3A_STREAM_1_STATE);
    assert_eq!(stream_1.position(), Some(0));

    let substream_0_1 = base.substream(0, 1).unwrap();
    assert_eq!(
        mrg_state_words(&substream_0_1),
        MRG32K3A_SUBSTREAM_0_1_STATE
    );
    assert_eq!(substream_0_1.position(), Some(0));

    let substream_1_1 = base.substream(1, 1).unwrap();
    assert_eq!(
        mrg_state_words(&substream_1_1),
        MRG32K3A_SUBSTREAM_1_1_STATE
    );
    assert_eq!(substream_1_1.position(), Some(0));
}

#[test]
fn xoshiro256_jump_prefixes_are_separated() {
    let base = Xoshiro256StarStar::from_seed(REFERENCE_SEED);
    let mut jump_once = base.clone();
    jump_once.jump().unwrap();
    let mut jump_twice = jump_once.clone();
    jump_twice.jump().unwrap();
    let mut long_jump_once = base.clone();
    long_jump_once.long_jump().unwrap();
    let mut long_jump_twice = long_jump_once.clone();
    long_jump_twice.long_jump().unwrap();

    assert_pairwise_disjoint(&[
        u64_prefix(base),
        u64_prefix(jump_once),
        u64_prefix(jump_twice),
        u64_prefix(long_jump_once),
        u64_prefix(long_jump_twice),
    ]);
}

#[test]
fn xoshiro128_jump_prefixes_are_separated() {
    let base = Xoshiro128StarStar::from_seed(REFERENCE_SEED);
    let mut jump_once = base.clone();
    jump_once.jump().unwrap();
    let mut jump_twice = jump_once.clone();
    jump_twice.jump().unwrap();
    let mut long_jump_once = base.clone();
    long_jump_once.long_jump().unwrap();
    let mut long_jump_twice = long_jump_once.clone();
    long_jump_twice.long_jump().unwrap();

    assert_pairwise_disjoint(&[
        u32_prefix(base),
        u32_prefix(jump_once),
        u32_prefix(jump_twice),
        u32_prefix(long_jump_once),
        u32_prefix(long_jump_twice),
    ]);
}
