//! Fixed-cost helpers for discrete rejection samplers.

#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)] // Needed when float methods resolve through RealScalar under no_std.
use scioxide_core::RealScalar as _;

/// Stirling remainders `ln(k!) - [(k + 1/2) ln k - k + ln sqrt(2 pi)]` for
/// `k = 0..=15`: the binary64 evaluation of that expression from `ln(k!)`
/// rounded once, tabulated so small counts need no logarithm (zero is `+inf`).
const STIRLING_SMALL: [f64; 16] = [
    f64::INFINITY,
    f64::from_bits(0x3fb4_c071_bcda_0a60),
    f64::from_bits(0x3fa5_2a9b_923e_a650),
    f64::from_bits(0x3f9c_579a_268d_8080),
    f64::from_bits(0x3f95_4a26_62fd_7900),
    f64::from_bits(0x3f91_0b4e_513f_cd00),
    f64::from_bits(0x3f8c_6b16_7beb_e000),
    f64::from_bits(0x3f88_5d4d_612e_4c00),
    f64::from_bits(0x3f85_5280_5e7b_3800),
    f64::from_bits(0x3f82_f487_1b12_a800),
    f64::from_bits(0x3f81_0f9d_4c07_4000),
    f64::from_bits(0x3f7f_0593_0880_1000),
    f64::from_bits(0x3f7c_7018_733a_a000),
    f64::from_bits(0x3f7a_4051_4701_0000),
    f64::from_bits(0x3f78_6076_c002_f000),
    f64::from_bits(0x3f76_c08f_6f19_5000),
];

/// Compare a uniform binary expansion with an exactly represented probability.
/// Every finite f64 probability terminates within 1,074 bits (17 words).
/// Unlike a single rounded uniform, this preserves subnormal probabilities.
#[inline]
pub(super) fn bernoulli<R: crate::BitGenerator + ?Sized>(
    mut p: f64,
    rng: &mut R,
) -> Result<bool, scioxide_core::SciOxideError> {
    const WORD_SCALE: f64 = 18_446_744_073_709_551_616.0;
    if p == 1.0 {
        return Ok(true);
    }
    // Exact Bernoulli(p): compare the binary expansion of a uniform variate
    // with that of p, 64 bits per raw word. The first word that differs from
    // p's current 64-bit digit decides the outcome; on a tie the next 64 bits
    // of p (the scaled remainder) are compared. Because p is an f64, its
    // expansion terminates after a few words, so the loop ends.
    while p > 0.0 {
        let scaled = p * WORD_SCALE;
        let threshold = scaled as u64;
        let word = rng.next_uniform_u64()?;
        let remainder = scaled - threshold as f64;
        if remainder == 0.0 || word != threshold {
            return Ok(word < threshold);
        }
        p = remainder;
    }
    Ok(false)
}

/// Loader's deviance `x ln(x / mean) + mean - x`, evaluated without
/// cancellation near the mean. The series ratio is at most 0.01 and 32
/// terms suffice for f64 rounding. Scale sums to avoid overflow.
// Loader's (2000) saddle-point deviance bd0(x, m) = x ln(x/m) + m - x. When
// x is close to m the closed form cancels badly, so it is expanded as the
// odd series in v = (x - m) / (x + m), which converges fast for |v| < 0.1.
pub(super) fn deviance(x: f64, mean: f64) -> f64 {
    if x == 0.0 {
        return mean;
    }
    let difference = x - mean;
    let half_sum = 0.5 * x + 0.5 * mean;
    if difference.abs() < 0.2 * half_sum {
        let v = (0.5 * difference) / half_sum;
        let mut term = 2.0 * x * v;
        let mut sum = difference * v;
        for j in 1..=32 {
            term *= v * v;
            let next = sum + term / f64::from(2 * j + 1);
            if next == sum {
                break;
            }
            sum = next;
        }
        sum
    } else {
        x * (x.ln() - mean.ln()) + mean - x
    }
}

/// `ln sqrt(2 pi)`, the correctly rounded half of `ln(2 pi)`.
const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;

/// Loader's deviance with a prepared `ln(mean)` for its closed form.
fn deviance_with_log(x: f64, mean: f64, log_mean: f64) -> f64 {
    if x == 0.0 {
        return mean;
    }
    let difference = x - mean;
    let half_sum = 0.5 * x + 0.5 * mean;
    if difference.abs() < 0.2 * half_sum {
        deviance(x, mean)
    } else {
        x * (x.ln() - log_mean) + mean - x
    }
}

/// Stable log Poisson mass, shared by analytics and PTRS acceptance.
pub(super) fn poisson_log_mass(k: f64, lambda: f64) -> f64 {
    PoissonMass::new(lambda).log_mass(k)
}

/// Loader's Poisson log mass with its rate logarithm prepared once.
#[derive(Debug, Clone, Copy)]
pub(super) struct PoissonMass {
    lambda: f64,
    log_lambda: f64,
}

impl PoissonMass {
    pub(super) fn new(lambda: f64) -> Self {
        Self {
            lambda,
            log_lambda: lambda.ln(),
        }
    }
    pub(super) fn log_mass(&self, k: f64) -> f64 {
        if k == 0.0 {
            return -self.lambda;
        }
        if self.lambda == 0.0 {
            return f64::NEG_INFINITY;
        }
        -deviance_with_log(k, self.lambda, self.log_lambda)
            - HALF_LOG_TWO_PI
            - 0.5 * k.ln()
            - factorial_stirling_correction(k)
    }
}

/// `ln C(n, k) = -ln(n + 1) - ln B(k + 1, n - k + 1)`: the one log binomial
/// coefficient, which never differences three large log-factorials.
/// Both `n` and `n + 1` must be exact counts, including at the endpoints.
pub(super) fn log_choose(n: usize, k: usize) -> Result<f64, scioxide_core::SciOxideError> {
    let successor = super::count_math::analytic_count(n.checked_add(1).ok_or_else(|| {
        scioxide_core::SciOxideError::not_implemented("binomial coefficient count overflow")
    })?)?;
    if k == 0 || k == n {
        return Ok(0.0);
    }
    let k = super::count_math::analytic_count(k.min(n - k))?;
    Ok(-successor.ln() - scioxide_special::ln_beta(k + 1.0, successor - k)?)
}

/// Stable log binomial mass using Loader's deviance/Stirling formulation.
pub(super) fn binomial_log_mass(n: f64, k: f64, p: f64) -> f64 {
    BinomialMass::new(n, p).log_mass(k)
}

/// Loader's binomial log mass with the constants of fixed `n` and `p` prepared.
#[derive(Debug, Clone, Copy)]
pub(super) struct BinomialMass {
    n: f64,
    success_mean: f64,
    failure_mean: f64,
    log_success_mean: f64,
    log_failure_mean: f64,
    log_p: f64,
    log_q: f64,
    stirling_n: f64,
}

impl BinomialMass {
    pub(super) fn new(n: f64, p: f64) -> Self {
        let (success_mean, failure_mean) = (n * p, n * (1.0 - p));
        Self {
            n,
            success_mean,
            failure_mean,
            log_success_mean: success_mean.ln(),
            log_failure_mean: failure_mean.ln(),
            log_p: p.ln(),
            log_q: (-p).ln_1p(),
            stirling_n: factorial_stirling_correction(n),
        }
    }
    pub(super) fn log_mass(&self, k: f64) -> f64 {
        let n = self.n;
        if k == 0.0 {
            return n * self.log_q;
        }
        if k == n {
            return n * self.log_p;
        }
        // Loader's saddle-point form of the binomial pmf: Stirling corrections
        // plus two deviance terms. Unlike ln C(n, k) + k ln p + (n-k) ln q it
        // does not subtract large nearly equal log-factorials, so relative
        // accuracy holds for huge n.
        let failures = n - k;
        let correction = self.stirling_n
            - factorial_stirling_correction(k)
            - factorial_stirling_correction(failures);
        correction
            - deviance_with_log(k, self.success_mean, self.log_success_mean)
            - deviance_with_log(failures, self.failure_mean, self.log_failure_mean)
            - HALF_LOG_TWO_PI
            - 0.5 * (k.ln() + (-k / n).ln_1p())
    }
}

// ln k! - [(k + 1/2) ln k - k + ln sqrt(2 pi)]. Small k use exact tabulated
// log-factorials; from 16 on the asymptotic series 1/(12k) - 1/(360k^3) +
// 1/(1260k^5) - 1/(1680k^7) is below double rounding error.
pub(super) fn factorial_stirling_correction(k: f64) -> f64 {
    if k < 16.0 {
        return STIRLING_SMALL[k as usize];
    }

    let inverse = k.recip();
    let inverse_squared = inverse * inverse;
    inverse
        * (1.0 / 12.0
            + inverse_squared
                * (-1.0 / 360.0 + inverse_squared * (1.0 / 1_260.0 - inverse_squared / 1_680.0)))
}

/// Stirling remainder for real positive arguments. Integer arguments use
/// the factorial table; fractional small shapes require the gamma function.
pub(super) fn real_stirling_correction(x: f64) -> Result<f64, scioxide_core::SciOxideError> {
    if x >= 16.0 || x == x.floor() {
        Ok(factorial_stirling_correction(x))
    } else {
        Ok(scioxide_special::ln_gamma(x + 1.0)? - ((x + 0.5) * x.ln() - x + HALF_LOG_TWO_PI))
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn raw_bernoulli_decodes_dyadic_probabilities_without_modulo_bias() {
        struct Word(u64);
        impl crate::BitGenerator for Word {
            #[inline]
            fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
                self.next_u32()
            }
            #[inline]
            fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
                self.next_u64()
            }
            fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
                Ok(self.0)
            }
            fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
                Ok(self.0 as u32)
            }
            fn fill_bytes(
                &mut self,
                output: &mut [u8],
            ) -> Result<(), scioxide_core::SciOxideError> {
                output.fill(0);
                Ok(())
            }
        }
        for numerator in [1, 3, 64, 127, 128, 255] {
            let successes = (0..256_u64)
                .filter(|&word| {
                    bernoulli(f64::from(numerator) / 256.0, &mut Word(word << 56)).unwrap()
                })
                .count();
            assert_eq!(successes, numerator as usize);
        }
    }
}
