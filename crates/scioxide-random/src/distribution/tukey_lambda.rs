//! Tukey lambda distribution.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::constant as number;
use crate::law::{argument, finite, log1mexp, logadd};
use crate::{Mean, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Tukey lambda distribution with shape `λ`, defined by its quantile function.
///
/// ```text
/// Q(p) = (p^λ − (1 − p)^λ) / λ,
/// ```
///
/// with the logistic quantile `Q(p) = ln(p / (1 − p))` at `λ = 0`. For `λ > 0`
/// the support is `[−1/λ, 1/λ]`. Draws and quantiles evaluate `Q` directly.
/// The density `1/Q′(p)` and the CDF need the root `p` of `Q(p) = x`. The mean
/// exists for `λ > −1` and the variance for `λ > −1/2`. The variance
/// `2 (1/(1 + 2λ) − B(1 + λ, 1 + λ)) / λ²`, with `B` the beta function, is
/// evaluated in `f64` from the series of `ln Γ` for `|λ| < 0.05`, with relative
/// error below 2·10⁻¹⁵, and from the closed form otherwise.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Quantile, SeedableRng, SplitMix64, TukeyLambda, Variance,
/// };
/// // lambda = 1 is the uniform law on [-1, 1].
/// let law = TukeyLambda::<f64>::new(1.0)?;
/// assert!((law.quantile(0.75)? - 0.5).abs() < 1e-15);
/// assert!((law.cdf(0.5)? - 0.75).abs() < 1e-14);
/// assert!((law.variance()? - 1.0 / 3.0).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(24);
/// assert!(law.sample(&mut rng)?.abs() <= 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TukeyLambda<S: RealScalar> {
    lambda: S,
}
impl<S: RealScalar> TukeyLambda<S> {
    /// Creates a Tukey lambda distribution with shape `λ = lambda`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lambda` is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if `lambda > 0` and `1/lambda` is not finite.
    pub fn new(lambda: S) -> Result<Self, SciOxideError> {
        argument(lambda, "lambda")?;
        if lambda > S::ZERO {
            finite(S::ONE / lambda)?;
        }
        Ok(Self { lambda })
    }
    fn quantile_at(&self, p: S) -> S {
        if p == S::ZERO {
            return if self.lambda > S::ZERO {
                -S::ONE / self.lambda
            } else {
                S::NEG_INFINITY
            };
        }
        if p == S::ONE {
            return if self.lambda > S::ZERO {
                S::ONE / self.lambda
            } else {
                S::INFINITY
            };
        }
        // Q(p) = (p^lambda - (1 - p)^lambda) / lambda, written with `expm1`
        // of lambda ln p so that small lambda approaches the logistic limit
        // (lambda = 0) continuously instead of cancelling.
        if self.lambda == S::ZERO {
            return p.ln() - (-p).ln_1p();
        }
        if p == S::HALF {
            return S::ZERO;
        }
        let log_p = p.ln();
        let log_q = (-p).ln_1p();
        let difference = (log_p - log_q).abs();
        let delta = self.lambda.abs() * difference;
        if delta == S::ZERO {
            return log_p - log_q;
        }
        // Subtract exponentials relative to the larger term, instead of
        // subtracting two expm1 values that have both rounded to -1.
        let log_magnitude = (self.lambda * log_p).max(self.lambda * log_q)
            + (-(-delta).exp_m1()).ln()
            - self.lambda.abs().ln();
        let magnitude = log_magnitude.exp();
        if p < S::HALF { -magnitude } else { magnitude }
    }
    // Tukey lambda has no closed-form CDF, only a quantile. For x <= 0 the CDF
    // is found by bisecting ln p over [ln(min_positive) - 60, ln(1/2)], which
    // brackets every representable lower-tail probability including
    // subnormals. Searching in log p keeps relative accuracy far in the tail;
    // 128 halvings exceed the resolution of any supported float.
    fn small_log_probability(&self, x: S) -> Result<S, SciOxideError> {
        if x == S::ZERO {
            return Ok(-S::LN_2);
        }
        let lower = S::MIN_POSITIVE.ln() - number::<S>(60.)?;
        let mut lo = lower;
        let mut hi = -S::LN_2;
        for _ in 0..128 {
            let mid = (lo + hi) * S::HALF;
            if mid == lo || mid == hi {
                break;
            }
            let value = self.quantile_at(mid.exp());
            if value < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok((lo + hi) * S::HALF)
    }
}
impl<S: RealScalar> Formula<S> for TukeyLambda<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        if self.lambda > S::ZERO {
            (Some(-S::ONE / self.lambda), Some(S::ONE / self.lambda))
        } else {
            (None, None)
        }
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let (low, high) = self.bounds();
        if low.is_some_and(|a| x < a) || high.is_some_and(|b| x > b) {
            return Ok(S::NEG_INFINITY);
        }
        if low == Some(x) || high == Some(x) {
            return Ok(if self.lambda < S::ONE {
                S::NEG_INFINITY
            } else if self.lambda == S::ONE {
                -S::LN_2
            } else {
                S::ZERO
            });
        }
        // f(x) = 1 / Q'(p) with Q'(p) = p^(lambda-1) + (1-p)^(lambda-1),
        // evaluated at the lower-tail p for -|x| by symmetry.
        let l = self.small_log_probability(-x.abs())?;
        let p = l.exp();
        Ok(-logadd(
            (self.lambda - S::ONE) * l,
            (self.lambda - S::ONE) * (-p).ln_1p(),
        )?)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        let x = if upper { -x } else { x };
        let (low, high) = self.bounds();
        if low.is_some_and(|a| x <= a) {
            return Ok(S::NEG_INFINITY);
        }
        if high.is_some_and(|b| x >= b) {
            return Ok(S::ZERO);
        }
        if x <= S::ZERO {
            self.small_log_probability(x)
        } else {
            log1mexp(self.small_log_probability(-x)?)
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        Ok(self.quantile_at(p))
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        Ok(-self.quantile_at(p))
    }
}
formula_law!(TukeyLambda);
impl<S: RealScalar> Mean<S> for TukeyLambda<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.lambda <= -S::ONE {
            Err(crate::law::undefined_moment(
                "Tukey-lambda mean does not exist",
            ))
        } else {
            Ok(S::ZERO)
        }
    }
}
impl<S: RealScalar> Variance<S> for TukeyLambda<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.lambda <= -S::HALF {
            return Err(crate::law::undefined_moment(
                "Tukey-lambda variance does not exist",
            ));
        }
        // V = 2 (1/(1 + 2l) - B(1 + l, 1 + l)) / l^2, evaluated in f64.
        let l = self.lambda.into_f64();
        let variance = if l.abs() < 0.05 {
            // B(1 + l, 1 + l) = exp(E) / (1 + 2l) with
            // E = sum_k (-1)^k zeta(k) (2 - 2^k) l^k / k (from the series of
            // ln Gamma(1 + x)), so V = -2 (E/l^2) (expm1(E)/E) / (1 + 2l)
            // without cancellation. Sixteen terms truncate below 2e-17
            // relative for |l| < 0.05.
            let scaled = TUKEY_VARIANCE_SERIES
                .iter()
                .rev()
                .fold(0.0, |sum, &c| sum * l + c);
            let e = scaled * l * l;
            let ratio = if e == 0.0 { 1.0 } else { e.exp_m1() / e };
            -2.0 * scaled * ratio / (1.0 + 2.0 * l)
        } else {
            let beta = scioxide_special::ln_beta(1.0 + l, 1.0 + l)?.exp();
            2.0 * (1.0 / (1.0 + 2.0 * l) - beta) / (l * l)
        };
        finite(S::try_from_f64_lossy(variance)?)
    }
}

/// Coefficients `(−1)ᵏ ζ(k) (2 − 2ᵏ) / k` for `k = 2, …, 17` of the series
/// `E/l² = Σ cₖ lᵏ⁻²` used by the Tukey lambda variance.
const TUKEY_VARIANCE_SERIES: [f64; 16] = [
    -1.644_934_066_848_226_4,
    2.404_113_806_319_188_5,
    -3.788_131_317_988_983_5,
    6.221_566_530_860_22,
    -10.512_544_973_839_308,
    18.150_286_992_874_612,
    -31.879_456_059_284_73,
    56.780_475_593_477_995,
    -102.301_645_578_063,
    186.091_919_080_366_2,
    -341.250_623_195_770_3,
    630.077_309_408_974_4,
    -1_170.214_526_210_609_4,
    2_184.466_816_943_389,
    -4_095.937_594_224_255_5,
    7_710.058_882_793_788,
];

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Reference values from the ln Γ series of E summed to 260 terms in
    // 50-digit decimal arithmetic.
    #[test]
    fn variance_is_accurate_across_the_series_switch() {
        for (lambda, expected, tolerance) in [
            (1e-8_f64, 3.289_868_019_816_817, 2e-15),
            (0.000_99, 3.278_621_087_743_555_3, 2e-15),
            (0.001, 3.278_507_756_495_721_5, 2e-15),
            (-0.001, 3.301_283_803_909_648_4, 2e-15),
            (0.0499, 2.783_712_445_402_324_8, 2e-15),
            (-0.0499, 3.935_255_845_869_075_4, 2e-15),
            (0.0501, 2.781_909_027_865_901_8, 1e-13),
            (-0.0501, 3.938_190_288_936_05, 1e-13),
            (0.2, 1.780_352_178_984_402_1, 1e-13),
            (-0.3, 13.354_712_585_021_803, 1e-13),
        ] {
            let got = TukeyLambda::new(lambda).unwrap().variance().unwrap();
            assert!(
                (got - expected).abs() <= tolerance * expected,
                "lambda = {lambda}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn f32_variance_near_zero_is_correctly_rounded() {
        for (lambda, expected) in [
            (0.000_976_562_5_f32, 3.278_773_385_259_155_4_f64),
            (-0.000_976_562_5, 3.301_015_613_648_665),
            (0.031_25, 2.959_278_032_546_158),
            (0.0625, 2.673_185_426_651_885),
        ] {
            let got = TukeyLambda::new(lambda).unwrap().variance().unwrap();
            assert_eq!(got, expected as f32, "lambda = {lambda}");
        }
    }
}
