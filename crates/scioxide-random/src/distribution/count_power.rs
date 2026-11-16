//! Finite power sums with constant storage and explicit work accounting.
use super::count_math::analytic_count;
use crate::util::neumaier;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)]
use scioxide_core::RealScalar as _;
use scioxide_core::SciOxideError;

// B_(2j)/(2j)!, j=1..8. Euler--Maclaurin's remainder after j-1
// corrections is bounded by the absolute jth correction for x^-a, a>=0.
const BERNOULLI: [f64; 8] = [
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30_240.0,
    -1.0 / 1_209_600.0,
    1.0 / 47_900_160.0,
    -691.0 / 1_307_674_368_000.0,
    1.0 / 74_724_249_600.0,
    -3617.0 / 10_670_622_842_880_000.0,
];

/// ln sum_{k=start}^end k^-a, for a>=-2. Scaling by start^-a
/// preserves small upper tails even when every unscaled mass underflows.
pub(super) fn power_sum(
    a: f64,
    start: usize,
    end: usize,
    left: &mut usize,
) -> Result<f64, SciOxideError> {
    if start > end {
        return Ok(f64::NEG_INFINITY);
    }
    let base = analytic_count(start)?;
    let last = analytic_count(end)?;
    if a == 0.0 {
        return Ok(((end - start + 1) as f64).ln());
    }
    let scale = -a * base.ln();
    let mut sum = (0.0, 0.0);
    let mut k = start;
    loop {
        crate::law::charge(left)?;
        let x = k as f64;
        let value = (-a * ((x - base) / base).ln_1p()).exp();
        sum = neumaier(sum, value);
        if k == end {
            return Ok(scale + (sum.0 + sum.1).ln());
        }
        k += 1;
        let x = k as f64;
        let value = (-a * ((x - base) / base).ln_1p()).exp();
        // The integral test bounds the entire unvisited decreasing tail.
        if a > 1.0 && value * (1.0 + x / (a - 1.0)) <= 4.0 * f64::EPSILON * (sum.0 + sum.1) {
            return Ok(scale + (sum.0 + sum.1).ln());
        }
        if end - k > 32 && k.is_multiple_of(32) {
            let ratio_log = ((last - x) / x).ln_1p();
            let exponent = 1.0 - a;
            let integral = if exponent == 0.0 {
                x * value * ratio_log
            } else {
                x * value * (exponent * ratio_log).exp_m1() / exponent
            };
            let endpoint = (-a * ((last - base) / base).ln_1p()).exp();
            let mut estimate = neumaier(sum, integral + 0.5 * (value + endpoint));
            let mut derivative = a / x;
            for (j, coefficient) in BERNOULLI.into_iter().enumerate() {
                crate::law::charge(left)?;
                let order = (2 * j + 1) as f64;
                let correction =
                    coefficient * derivative * value * -(-(a + order) * ratio_log).exp_m1();
                // Integral of |f^(2j+2)| times |B_(2j+2)|/(2j+2)!
                // bounds the remainder before this correction. The same
                // formula covers -2<=a<0; vanishing derivatives are exact.
                if correction.abs() <= 4.0 * f64::EPSILON * (estimate.0 + estimate.1).abs() {
                    return Ok(scale + (estimate.0 + estimate.1).ln());
                }
                if !correction.is_finite() {
                    break;
                }
                estimate = neumaier(estimate, correction);
                derivative *= (a + order) / x * ((a + order + 1.0) / x);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn euler_maclaurin_and_integral_tail_bounds_match_direct_positive_sums() {
        for a in [
            -2.0_f64,
            -1.0,
            -0.5,
            0.0,
            0.25,
            1.0_f64.next_down(),
            1.0,
            1.0_f64.next_up(),
            1.1,
            3.0,
            20.0,
            200.0,
        ] {
            for (start, end) in [(1, 40), (1, 10_000), (73, 10_000), (9_000, 10_000)] {
                let result = power_sum(a, start, end, &mut 65_536).unwrap();
                let (sum, correction) = (start..=end)
                    .map(|k| (-a * (k as f64).ln()).exp())
                    .fold((0.0, 0.0), neumaier);
                let expected = (sum + correction).ln();
                assert!(
                    (result - expected).abs() <= 8e-14 * expected.abs().max(1.0),
                    "a={a}, {start}..={end}: {result} vs {expected}"
                );
            }
        }
        assert_eq!(power_sum(f64::MAX, 1, 100_000, &mut 1).unwrap(), 0.0);
        let error = power_sum(1.1, 1, 100_000, &mut 1).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
    }
}
