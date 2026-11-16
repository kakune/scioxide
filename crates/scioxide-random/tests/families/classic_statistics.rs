//! Fixed-seed empirical battery: uniform bins, moments, serial correlation,
//! median runs and bit balance. These checks do not certify randomness.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use scioxide_random::*;

fn battery<B: BitGenerator + Rng<f64>>(name: &str, seed: u64, generator: B) {
    let mut rng = generator;
    let mut bins = [0_u32; 32];
    let mut bits = [0_u32; 24];
    let (mut sum, mut square, mut cross, mut previous, mut runs) = (0.0, 0.0, 0.0, 0.0, 0_u32);
    const N: u32 = 65536;
    for i in 0..N {
        let x: f64 = rng.next_float().unwrap();
        assert!((0.0..1.0).contains(&x));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let fraction = (x * 16777216.0) as u32;
        bins[(fraction >> 19) as usize] += 1;
        for (bit, count) in bits.iter_mut().enumerate() {
            *count += (fraction >> bit) & 1;
        }
        sum += x;
        square += x * x;
        if i > 0 {
            cross += previous * x;
            runs += u32::from((previous >= 0.5) != (x >= 0.5));
        }
        previous = x;
    }
    let n = f64::from(N);
    let mean = sum / n;
    let variance = square / n - mean * mean;
    let chi = bins
        .into_iter()
        .map(|v| (f64::from(v) - n / 32.0).powi(2) / (n / 32.0))
        .sum::<f64>();
    let serial = (cross / (n - 1.0) - mean * mean) / variance;
    assert!(chi < 100.0, "{name}/{seed}: bins {chi}");
    assert!((mean - 0.5).abs() < 0.01, "{name}/{seed}: mean {mean}");
    assert!(
        (variance - 1.0 / 12.0).abs() < 0.003,
        "{name}/{seed}: variance {variance}"
    );
    assert!(
        serial.abs() < 0.04,
        "{name}/{seed}: lag-one correlation {serial}"
    );
    assert!(
        (f64::from(runs) - (n - 1.0) / 2.0).abs() < 8.0 * (n / 4.0).sqrt(),
        "{name}/{seed}: runs {runs}"
    );
    assert!(
        bits.into_iter()
            .all(|count| (f64::from(count) - n / 2.0).abs() < 8.0 * (n / 4.0).sqrt()),
        "{name}/{seed}: bit balance"
    );
    eprintln!("{name}\t{seed}\t{mean:.8}\t{variance:.8}\t{chi:.6}\t{serial:.8}\t{runs}");
}
#[test]
fn fixed_seed_classic_battery() {
    for seed in [1, 42, 2026] {
        battery("mt19937", seed, Mt19937::from_seed(seed));
        battery("taus2", seed, Taus2::from_seed(seed));
        battery("taus113", seed, Taus113::from_seed(seed));
        battery("mrg", seed, Mrg::from_seed(seed));
        battery("cmrg", seed, Cmrg::from_seed(seed));
        battery("mrg32k3a", seed, Mrg32k3a::from_seed(seed));
        for level in 0..=4 {
            battery(
                &format!("ranlux{level}"),
                seed,
                Ranlux::with_luxury(seed, RanluxLuxury::try_from(level).unwrap()),
            );
        }
        for level in 0..=2 {
            battery(
                &format!("ranlxs{level}"),
                seed,
                Ranlxs::with_luxury(seed, RanlxsLuxury::try_from(level).unwrap()),
            );
        }
        for level in 1..=2 {
            battery(
                &format!("ranlxd{level}"),
                seed,
                Ranlxd::with_luxury(seed, RanlxdLuxury::try_from(level).unwrap()),
            );
        }
    }
}
#[test]
fn fixed_seed_gfsr_battery() {
    for seed in [1, 42, 2026] {
        battery("gfsr4", seed, Gfsr4::from_seed(seed));
    }
}
