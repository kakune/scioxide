//! Classic families: independent streams, portable replay and malformed states.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_random::*;

fn reference<B: BitGenerator + SerializableState>(mut engine: B, fields: &[&str], wide: bool) {
    let mut position = 0;
    for field in fields {
        let (index, value) = field.split_once(':').unwrap();
        if index == "crc" {
            let bytes = engine.export_state().unwrap();
            let actual = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
            assert_eq!(
                u64::from(actual),
                value.parse::<u64>().unwrap(),
                "independent checkpoint CRC"
            );
            continue;
        }
        let index: usize = index.parse().unwrap();
        let value: u64 = value.parse().unwrap();
        loop {
            let actual = if wide {
                engine.next_u64().unwrap()
            } else {
                u64::from(engine.next_u32().unwrap())
            };
            if position == index {
                assert_eq!(actual, value, "native output {index}");
                position += 1;
                break;
            }
            position += 1;
        }
    }
}
#[test]
fn gsl_28_comparison_streams() {
    run_reference(include_str!("../data/classic_gsl.tsv"));
}
#[test]
fn independent_recurrence_seed_and_checkpoint_vectors() {
    run_reference(include_str!("../data/classic_reference.tsv"));
}
fn run_reference(table: &str) {
    assert!(!table.is_empty());
    for row in table.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = row.split('\t').collect();
        let seed = fields[1].parse().unwrap();
        let values = &fields[2..];
        match fields[0] {
            "mt19937" => reference(Mt19937::from_seed(seed), values, false),
            "ranlux0" => reference(
                Ranlux::with_luxury(seed, RanluxLuxury::Level0),
                values,
                false,
            ),
            "ranlux1" => reference(
                Ranlux::with_luxury(seed, RanluxLuxury::Level1),
                values,
                false,
            ),
            "ranlux2" => reference(
                Ranlux::with_luxury(seed, RanluxLuxury::Level2),
                values,
                false,
            ),
            "ranlux" | "ranlux3" => reference(Ranlux::from_seed(seed), values, false),
            "ranlux389" | "ranlux4" => reference(
                Ranlux::with_luxury(seed, RanluxLuxury::Level4),
                values,
                false,
            ),
            "ranlxs0" => reference(
                Ranlxs::with_luxury(seed, RanlxsLuxury::Level0),
                values,
                false,
            ),
            "ranlxs1" => reference(Ranlxs::from_seed(seed), values, false),
            "ranlxs2" => reference(
                Ranlxs::with_luxury(seed, RanlxsLuxury::Level2),
                values,
                false,
            ),
            "ranlxd1" => reference(Ranlxd::from_seed(seed), values, true),
            "ranlxd2" => reference(
                Ranlxd::with_luxury(seed, RanlxdLuxury::Level2),
                values,
                true,
            ),
            "taus2" => reference(Taus2::from_seed(seed), values, false),
            "taus113" => reference(Taus113::from_seed(seed), values, false),
            "gfsr4" => reference(Gfsr4::from_seed(seed), values, false),
            "mrg" => reference(Mrg::from_seed(seed), values, false),
            "cmrg" => reference(Cmrg::from_seed(seed), values, false),
            name => panic!("unknown reference engine {name}"),
        }
    }
}
fn replay<B: SerializableState + Clone + Rng<f32> + Rng<f64>>(mut engine: B) {
    let root = engine.clone();
    for boundary in [0, 1, 11, 12, 23, 24, 25, 623, 624, 625, 16383, 16384] {
        while engine.position().unwrap() < boundary {
            engine.next_u32().unwrap();
        }
        let bytes = engine.export_state().unwrap();
        let mut restored = B::import_state(&bytes).unwrap();
        let mut expected = engine.clone();
        assert_eq!(restored.stream_identity(), root.stream_identity());
        assert_eq!(restored.export_state().unwrap(), bytes);
        for _ in 0..25 {
            assert_eq!(restored.next_u64().unwrap(), expected.next_u64().unwrap());
            let a: f32 = restored.next_float().unwrap();
            let b: f32 = expected.next_float().unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
            assert!((0.0..1.0).contains(&a));
            let a: f64 = restored.next_float().unwrap();
            let b: f64 = expected.next_float().unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
            assert!((0.0..1.0).contains(&a));
            let mut a = [0; 17];
            let mut b = [0; 17];
            restored.fill_bytes(&mut a).unwrap();
            expected.fill_bytes(&mut b).unwrap();
            assert_eq!(a, b);
        }
        engine.restore_state(&bytes).unwrap();
        assert_eq!(engine.export_state().unwrap(), bytes);
    }
}
#[test]
fn portable_replay_all_families_and_luxuries() {
    replay(Mt19937::from_seed(0));
    replay(Mt19937::from_seed_words(&[0, 1, u32::MAX]).unwrap());
    replay(Taus2::from_seed(u64::MAX));
    replay(Taus113::from_seed(0));
    replay(Mrg::from_seed(0));
    replay(Cmrg::from_seed(u64::MAX));
    for level in 0..=4 {
        replay(Ranlux::with_luxury(
            0,
            RanluxLuxury::try_from(level).unwrap(),
        ));
    }
    for level in 0..=2 {
        replay(Ranlxs::with_luxury(
            42,
            RanlxsLuxury::try_from(level).unwrap(),
        ));
    }
    for level in 1..=2 {
        replay(Ranlxd::with_luxury(
            42,
            RanlxdLuxury::try_from(level).unwrap(),
        ));
    }
}
#[test]
fn gfsr_portable_replay() {
    replay(Gfsr4::from_seed(42));
}

fn crc(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |mut c, &b| {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = (c >> 1) ^ if c & 1 == 0 { 0 } else { 0x82f6_3b78 };
        }
        c
    })
}
fn repair(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let c = crc(&bytes[..end]);
    bytes[end..].copy_from_slice(&c.to_le_bytes());
}
fn invalid<B: SerializableState + SeedableRng>() {
    let mut engine = B::from_seed(42);
    let saved = engine.export_state().unwrap();
    for at in [0, 4, 6, 7, 8, 9, 10, 12, 16, saved.len() - 1] {
        let mut bad = saved.clone();
        bad[at] ^= 0xff;
        assert!(B::import_state(&bad).is_err());
        repair(&mut bad);
        if at != saved.len() - 1 {
            assert!(engine.restore_state(&bad).is_err());
        }
        assert_eq!(engine.export_state().unwrap(), saved);
    }
    for length in [0, 15, saved.len() - 1, saved.len() + 1] {
        let mut bad = saved.clone();
        bad.resize(length, 0);
        assert!(B::import_state(&bad).is_err());
        let unchanged = bad.clone();
        assert!(engine.export_state_into(&mut bad).is_err());
        assert_eq!(bad, unchanged);
        assert_eq!(engine.export_state().unwrap(), saved);
    }
    let mut bad = saved.clone();
    bad[16..saved.len() - 20].fill(0);
    repair(&mut bad);
    assert!(B::import_state(&bad).is_err());
    assert!(
        engine
            .restore_state(&B::from_seed(7).export_state().unwrap())
            .is_err()
    );
    assert_eq!(engine.export_state().unwrap(), saved);
    let mut overflow = saved;
    let end = overflow.len() - 4;
    overflow[end - 16..end].copy_from_slice(&u128::MAX.to_le_bytes());
    repair(&mut overflow);
    let mut overflow = B::import_state(&overflow).unwrap();
    overflow.next_u32().unwrap();
    assert_eq!(overflow.position(), None);
    assert_eq!(
        B::import_state(&overflow.export_state().unwrap())
            .unwrap()
            .position(),
        None
    );
}
#[test]
fn invalid_states_and_counter_overflow() {
    invalid::<Mt19937>();
    invalid::<Ranlux>();
    invalid::<Ranlxs>();
    invalid::<Ranlxd>();
    invalid::<Taus2>();
    invalid::<Taus113>();
    invalid::<Mrg>();
    invalid::<Cmrg>();
    assert!(RanluxLuxury::try_from(5).is_err());
    assert!(RanlxsLuxury::try_from(3).is_err());
    assert!(RanlxdLuxury::try_from(0).is_err());
}
#[test]
fn invalid_gfsr_states() {
    invalid::<Gfsr4>();
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decoders<B: SeedableRng + BitGenerator + Rng<f32> + Rng<f64>>(bits: u32, divisor: f64) {
    let mut raw = B::from_seed(42);
    let mut float = B::from_seed(42);
    for _ in 0..128 {
        let word = if bits > 32 {
            raw.next_u64().unwrap()
        } else {
            u64::from(raw.next_u32().unwrap())
        };
        let actual: f64 = float.next_float().unwrap();
        assert_eq!(actual.to_bits(), (word as f64 / divisor).to_bits());
        let word = if bits > 32 {
            raw.next_u64().unwrap()
        } else {
            u64::from(raw.next_u32().unwrap())
        };
        let actual: f32 = float.next_float().unwrap();
        let expected = (((word as f64 / divisor) * 16777216.0).floor() / 16777216.0) as f32;
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let mut raw = B::from_seed(42);
    let low = if bits > 32 {
        raw.next_u64().unwrap()
    } else {
        u64::from(raw.next_u32().unwrap())
    };
    let expected = if bits > 32 {
        low
    } else {
        low | (u64::from(raw.next_u32().unwrap()) << 32)
    };
    assert_eq!(B::from_seed(42).next_u64().unwrap(), expected);
    for length in 0..=17 {
        let mut raw = B::from_seed(42);
        let mut expected = Vec::new();
        while expected.len() < length {
            let word = if bits > 32 {
                raw.next_u64().unwrap()
            } else {
                u64::from(raw.next_u32().unwrap())
            };
            expected.extend_from_slice(&word.to_le_bytes()[..bits.div_ceil(8) as usize]);
        }
        expected.truncate(length);
        let mut actual = vec![0; length];
        let mut engine = B::from_seed(42);
        engine.fill_bytes(&mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            engine.position(),
            Some(length.div_ceil(bits.div_ceil(8) as usize) as u128)
        );
    }
}
#[test]
fn native_composition_bytes_and_both_real_decoders() {
    decoders::<Mt19937>(32, 4294967296.0);
    decoders::<Taus2>(32, 4294967296.0);
    decoders::<Taus113>(32, 4294967296.0);
    decoders::<Ranlux>(24, 16777216.0);
    decoders::<Ranlxs>(24, 16777216.0);
    decoders::<Ranlxd>(48, 281474976710656.0);
    decoders::<Mrg>(32, 2147483647.0);
    decoders::<Cmrg>(32, 2147483647.0);
}
#[test]
fn gfsr_native_composition_and_decoders() {
    decoders::<Gfsr4>(32, 4294967296.0);
}

#[test]
fn checksummed_recurrence_metadata_is_validated() {
    fn corrupt<B: SeedableRng + SerializableState>(index: usize, value: u64) {
        let engine = B::from_seed(42);
        let mut bytes = engine.export_state().unwrap();
        let words = (bytes.len() - 36) / 16;
        for base in [16, 16 + 8 * words] {
            bytes[base + 8 * index..base + 8 * (index + 1)].copy_from_slice(&value.to_le_bytes());
            repair(&mut bytes);
            assert!(B::import_state(&bytes).is_err());
            bytes = engine.export_state().unwrap();
        }
    }
    corrupt::<Mt19937>(624, 625);
    corrupt::<Mt19937>(1, 1 << 32);
    corrupt::<Ranlux>(24, 2);
    corrupt::<Ranlux>(25, 24);
    corrupt::<Ranlux>(26, 24);
    corrupt::<Ranlux>(27, 5);
    corrupt::<Ranlxs>(12, 2);
    corrupt::<Ranlxs>(13, 12);
    corrupt::<Ranlxs>(14, 24);
    corrupt::<Ranlxs>(15, 23);
    corrupt::<Ranlxs>(16, 3);
    corrupt::<Ranlxd>(0, 1 << 48);
    corrupt::<Ranlxd>(14, 12);
    corrupt::<Ranlxd>(16, 0);
    corrupt::<Taus2>(0, 1);
    corrupt::<Taus113>(3, 127);
    corrupt::<Mrg>(0, 2147483647);
    corrupt::<Cmrg>(3, 2145483479);
    let mut bytes = Ranlux::from_seed(42).export_state().unwrap();
    bytes[16 + 28 * 8 + 27 * 8..16 + 28 * 8 + 28 * 8].copy_from_slice(&4_u64.to_le_bytes());
    repair(&mut bytes);
    assert!(Ranlux::import_state(&bytes).is_err());
}
