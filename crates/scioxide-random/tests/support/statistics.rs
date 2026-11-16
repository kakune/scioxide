//! Deterministic goodness of fit under the IID sampling model. Pearson spends
//! alpha=1e-7 and DKW spends 9e-7, hence a pair spends at most 1e-6.
//! The default 32768 draws detect a .03 CDF discrepancy at a tested cutpoint
//! with probability >.99 (including the binary32 reference allowance).
//! Seeds stay at call sites. Cutpoints and expectations are prepared once;
//! neither reference inversion nor tail evaluation belongs in the draw loop.
#![allow(dead_code, clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_core::RealScalar;
use scioxide_random::{Cdf, Quantile};

pub const DRAWS: usize = 32_768;
const BINS: usize = 32;
pub const PEARSON_ALPHA: f64 = 1e-7;
pub const DKW_ALPHA: f64 = 9e-7;

/// Absolute CDF reference allowance, separate from sampling uncertainty.
/// Covers the retained family fixture tolerances (3e-3 for f32, 3e-6 for f64).
pub fn reference_error<S: RealScalar>() -> f64 {
    if S::MANTISSA_DIGITS == 24 { 3e-3 } else { 3e-6 }
}

/// Count samples in intervals ending at the supplied inclusive boundaries.
pub fn histogram<T: PartialOrd>(
    boundaries: &[T],
    draws: usize,
    mut draw: impl FnMut() -> T,
) -> Vec<usize> {
    let mut counts = vec![0; boundaries.len() + 1];
    assert!(boundaries.iter().all(|x| x.partial_cmp(x).is_some()));
    assert!(boundaries.windows(2).all(|w| w[0] < w[1]));
    for _ in 0..draws {
        let value = draw();
        assert!(value.partial_cmp(&value).is_some(), "unordered sample");
        counts[boundaries.partition_point(|boundary| value > *boundary)] += 1;
    }
    counts
}

/// Pearson statistic for observed counts and expected counts in the same order.
/// Zero expected counts require zero observations.
pub fn pearson(counts: &[usize], expected: impl IntoIterator<Item = f64>) -> f64 {
    let expected: Vec<_> = expected.into_iter().collect();
    assert_eq!(counts.len(), expected.len());
    counts
        .iter()
        .zip(expected)
        .map(|(&observed, expected)| {
            if expected == 0.0 {
                assert_eq!(observed, 0);
                0.0
            } else {
                let residual = observed as f64 - expected;
                residual * residual / expected
            }
        })
        .sum()
}

/// Check a categorical histogram, retaining zero-probability categories.
///
/// Bernstein gives |N_i-n p_i| <= sqrt(2 n p_i(1-p_i)t)+2t/3 except
/// with probability 2 exp(-t). A union bound at t=ln(2k/alpha) therefore
/// bounds Pearson by the sum of these squared limits / (n p_i), without a
/// chi-square approximation. Check individual bands as well, to retain power
/// against missing rare categories; both checks use the same Bernstein event.
/// See Vershynin, High-Dimensional Probability,
/// Bernstein inequality for bounded distributions. A CDF reference error e
/// permits at most 2e error per bin; use the maximal Bernoulli variance over
/// that interval as well as the deterministic n*2e displacement.
///
/// DKW/Massart bounds all cumulative counts simultaneously by
/// sqrt(ln(2/alpha)/(2n)); checking only selected cutpoints retains that bound.
/// At n=32768, the statistical band is .014933. If an alternative's true CDF
/// differs from the null by delta=.03 at a cutpoint, the miss probability is
/// <=2 exp(-2n(delta-band-2e)^2), <.01 even with e=.00301. Two e terms cover
/// displacement of the reference from the null and the numerical acceptance
/// allowance. The extra .00001 covers discrete PMF/CDF roundoff.
pub fn assert_fit(label: &str, counts: &[usize], probabilities: &[f64], error: f64) {
    assert_eq!(counts.len(), probabilities.len());
    assert!(!counts.is_empty());
    assert!(error.is_finite() && (0.0..0.01).contains(&error));
    assert!(
        probabilities
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
    );
    let total: f64 = probabilities.iter().sum();
    assert!(
        (total - 1.0).abs() <= 1e-10,
        "{label}: probability total {total}"
    );
    let n = counts.iter().sum::<usize>() as f64;
    assert!(n > 0.0);
    let t = (2.0 * counts.len() as f64 / PEARSON_ALPHA).ln();
    // A numerically rounded zero tail can still contain up to 2*error mass.
    // Give it a positive Pearson denominator; exact zeros remain impossible.
    let pearson_probabilities: Vec<_> = probabilities
        .iter()
        .map(|&p| if p == 0.0 { 2.0 * error } else { p })
        .collect();
    let limit: f64 = pearson_probabilities
        .iter()
        .zip(counts)
        .enumerate()
        .map(|(i, (&p, &observed))| {
            if p == 0.0 {
                return 0.0;
            }
            let q = 0.5_f64.clamp((p - 2.0 * error).max(0.0), (p + 2.0 * error).min(1.0));
            let bound = (2.0 * n * q * (1.0 - q) * t).sqrt() + 2.0 * t / 3.0 + 2.0 * n * error;
            assert!(
                (observed as f64 - n * p).abs() <= bound,
                "{label}: Pearson bin {i}: {observed}, expected {}, bound {bound}",
                n * p
            );
            bound * bound / (n * p)
        })
        .sum();
    let statistic = pearson(counts, pearson_probabilities.iter().map(|p| n * p));
    assert!(
        statistic <= limit,
        "{label}: Pearson {statistic} > {limit}; counts={counts:?}"
    );
    let band = ((2.0 / DKW_ALPHA).ln() / (2.0 * n)).sqrt() + error;
    let mut observed = 0;
    let mut expected = 0.0;
    for (&count, &p) in counts.iter().zip(probabilities).take(counts.len() - 1) {
        observed += count;
        expected += p;
        assert!(
            (observed as f64 / n - expected).abs() <= band,
            "{label}: CDF {} != {expected}, band {band}; counts={counts:?}",
            observed as f64 / n
        );
    }
}

/// Prepare strictly increasing CDF cuts, merging ties from discrete quantiles.
pub fn cdf_fit<T: PartialOrd + Copy>(
    label: &str,
    cuts: &[(T, f64)],
    draws: usize,
    error: f64,
    draw: impl FnMut() -> T,
) {
    let mut boundaries = Vec::new();
    let mut probabilities = Vec::new();
    let mut previous = 0.0;
    for &(cut, p) in cuts {
        assert!(
            p.is_finite() && p >= previous && p <= 1.0,
            "{label}: invalid CDF cut {p}"
        );
        if boundaries.last() == Some(&cut) {
            assert!(
                (p - previous).abs() <= error,
                "{label}: inconsistent tied cut"
            );
            continue;
        }
        boundaries.push(cut);
        probabilities.push(p - previous);
        previous = p;
    }
    probabilities.push(1.0 - previous);
    let counts = histogram(&boundaries, draws, draw);
    assert_fit(label, &counts, &probabilities, error);
}

/// Continuous equiprobable-bin test against fixture-validated forward/inverse APIs.
pub fn continuous<S: RealScalar, D: Quantile<S> + Cdf<S> + ?Sized>(
    label: &str,
    law: &D,
    draw: impl FnMut() -> S,
) {
    continuous_with_error(label, law, reference_error::<S>(), draw);
}

/// Same test with an explicit numerical allowance for a configured inversion
/// sampler. This changes numerical accuracy, not draw count or statistical alpha.
pub fn continuous_with_error<S: RealScalar, D: Quantile<S> + Cdf<S> + ?Sized>(
    label: &str,
    law: &D,
    error: f64,
    mut draw: impl FnMut() -> S,
) {
    let cuts: Vec<_> = (1..BINS)
        .map(|i| {
            let result = law.quantile(S::try_from_f64_lossy(i as f64 / BINS as f64).unwrap());
            assert!(
                result.is_ok(),
                "{label}: quantile cut {i}/{BINS}: {result:?}"
            );
            let x = result.unwrap();
            assert!(x.is_finite(), "{label}: nonfinite quantile");
            (x, law.cdf(x).unwrap().into_f64())
        })
        .collect();
    cdf_fit(label, &cuts, DRAWS, error, || {
        let x = draw();
        assert!(x.is_finite(), "{label}: nonfinite sample");
        x
    });
}

/// Group a finite central count window by PMF, retaining both outside tails.
/// Bounds are inclusive and need not be the complete support. No tail is
/// discarded. CDF checks independently validate PMF sums; only their scalar
/// roundoff discrepancy is normalized out of the total. The measured cumulative
/// PMF/CDF discrepancy must be <=1e-5 and is added to the numerical allowance.
pub fn discrete(
    label: &str,
    lo: i64,
    hi: i64,
    pmf: impl Fn(i64) -> f64,
    cdf: impl Fn(i64) -> f64,
    error: f64,
    draw: impl FnMut() -> i64,
) {
    assert!(lo <= hi);
    let lower = cdf(lo - 1);
    let mut boundaries = vec![lo - 1];
    let mut probabilities = vec![lower];
    let mut mass = 0.0;
    let mut cumulative = lower;
    for k in lo..=hi {
        let p = pmf(k);
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "{label}: invalid PMF at {k}"
        );
        mass += p;
        if mass >= 1.0 / BINS as f64 || k == hi {
            boundaries.push(k);
            probabilities.push(mass);
            cumulative += mass;
            assert!(
                (cumulative - cdf(k)).abs() <= error + 1e-10,
                "{label}: PMF/CDF at {k}"
            );
            mass = 0.0;
        }
    }
    probabilities.push(1.0 - cdf(hi));
    let total: f64 = probabilities.iter().sum();
    assert!(
        (total - 1.0).abs() <= error + 1e-10,
        "{label}: PMF total {total}"
    );
    for p in &mut probabilities {
        *p /= total;
    }
    let mut sum = 0.0;
    let gap = boundaries
        .iter()
        .zip(&probabilities)
        .map(|(&k, &p)| {
            sum += p;
            (sum - cdf(k)).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(gap <= 1e-5, "{label}: PMF/CDF roundoff {gap}");
    let counts = histogram(&boundaries, DRAWS, draw);
    assert_fit(label, &counts, &probabilities, error + gap);
}

/// Lehmer rank, requiring exactly the permutation of 0..len.
pub fn permutation_rank(permutation: &[usize]) -> usize {
    assert!(
        permutation
            .iter()
            .enumerate()
            .all(|(i, &x)| { x < permutation.len() && !permutation[..i].contains(&x) })
    );
    permutation.iter().enumerate().fold(0, |rank, (i, x)| {
        rank * (permutation.len() - i) + permutation[i + 1..].iter().filter(|y| *y < x).count()
    })
}
