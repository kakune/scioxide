//! Special-function compositions shared by scalar laws.
use crate::distribution::LawOptions;
use crate::law::{constant as c, finite};
use crate::{LogCdf, LogSurvival};
use scioxide_core::{RealScalar, SciOxideError};

pub(super) fn log_k<S: RealScalar>(p: S, x: S, left: &mut usize) -> Result<S, SciOxideError> {
    // ln K_p(x) = ln(e^x K_p(x)) - x: the exponentially scaled Bessel function
    // stays representable where K_p(x) itself underflows for large x.
    finite(scaled_log_k_with_work(p, x, left)? - x)
}

pub(super) fn scaled_log_k<S: RealScalar>(
    p: S,
    x: S,
    options: LawOptions<S>,
) -> Result<S, SciOxideError> {
    scaled_log_k_with_work(p, x, &mut options.work_budget().get())
}

pub(super) fn scaled_log_k_with_work<S: RealScalar>(
    p: S,
    x: S,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let value = crate::law::special(left, |budget| {
        scioxide_special::bessel_k_scaled_with_options(
            p.into_f64(),
            x.into_f64(),
            scioxide_special::BesselOptions::default().with_iteration_budget(budget),
        )
    })?;
    S::try_from_f64_lossy(RealScalar::ln(value))
}

pub(crate) fn normal_log(x: f64, upper: bool) -> Result<f64, SciOxideError> {
    if x.is_infinite() {
        return Ok(if (x > 0.0) == upper {
            f64::NEG_INFINITY
        } else {
            0.0
        });
    }
    let n = crate::Normal::<f64>::standard();
    if upper { n.log_sf(x) } else { n.log_cdf(x) }
}
pub(crate) fn normal_log_interval(a: f64, b: f64) -> Result<f64, SciOxideError> {
    // Use a centered expansion for narrow finite intervals; otherwise delegate
    // stable tail selection to the shared logarithmic interval evaluator.
    if !a.is_finite() || !b.is_finite() || a >= b {
        return crate::log_interval_probability(&crate::Normal::<f64>::standard(), a, b);
    }
    let width = b - a;
    let middle = a * 0.5 + b * 0.5;
    if width * (1.0 + middle.abs()) <= f64::SQRT_EPSILON.sqrt() {
        // Centered Gaussian integral: phi(m) w [1+(m²-1)w²/24+O(w⁴)].
        // This threshold puts the next Hermite term below working precision
        // and avoids subtracting nearly identical normal tails.
        let correction = (middle * middle - 1.0) * width * width / 24.0;
        return Ok(normal_log_density(middle) + width.ln() + correction.ln_1p());
    }
    crate::log_interval_probability(&crate::Normal::<f64>::standard(), a, b)
}

/// Natural logarithm of 2 pi in the working precision.
pub(crate) fn log_two_pi<S: RealScalar>() -> S {
    S::TAU.ln()
}

pub(crate) fn normal_log_density<S: RealScalar>(z: S) -> S {
    -z * (z * S::HALF) - log_two_pi::<S>() * S::HALF
}

/// `e^x - 1 - x` without cancellation near zero.
pub(super) fn expm1_minus<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    // e^x - 1 - x. Near zero both `expm1(x)` and `x` agree to leading order and
    // their difference cancels, so the Taylor series from x^2/2 is summed
    // instead; twelve terms exceed double precision for |x| <= 0.01.
    if x.abs() > c(0.01)? {
        return Ok(x.exp_m1() - x);
    }
    let mut term = x * x * S::HALF;
    let mut sum = term;
    for k in 3..=12 {
        term = term * x / S::try_from_usize_exact(k)?;
        sum += term;
    }
    Ok(sum)
}

/// Leading term `ln(x^a / (a B(a, b)))` of the regularized incomplete beta
/// `I_x(a, b)` from `ln x`, when its relative correction, bounded by
/// `(a + b) x`, is below rounding. It stays finite after `x` underflows.
pub(super) fn beta_leading_log<S: RealScalar>(
    a: S,
    b: S,
    log_x: S,
    log_beta: impl FnOnce() -> Result<S, SciOxideError>,
) -> Result<Option<S>, SciOxideError> {
    Ok(if log_x < (S::EPSILON / (a + b)).ln() {
        Some(a * log_x - a.ln() - log_beta()?)
    } else {
        None
    })
}

// Gauss-Legendre on [0,1]. For |shift| <= base/8 the nearest gamma pole
// is at least four panel widths away even in the second-difference integral.
// Eight nodes resolve these analytic integrands to binary64 roundoff.
const GAUSS: [(f64, f64); 8] = [
    (0.019_855_071_751_231_884, 0.050_614_268_145_188_13),
    (0.101_666_761_293_186_64, 0.111_190_517_226_687_24),
    (0.237_233_795_041_835_5, 0.156_853_322_938_943_65),
    (0.408_282_678_752_175_1, 0.181_341_891_689_181),
    (0.591_717_321_247_824_8, 0.181_341_891_689_181),
    (0.762_766_204_958_164_5, 0.156_853_322_938_943_65),
    (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
    (0.980_144_928_248_768_1, 0.050_614_268_145_188_13),
];

pub(super) fn average(f: impl Fn(f64) -> Result<f64, SciOxideError>) -> Result<f64, SciOxideError> {
    GAUSS
        .iter()
        .try_fold(0.0, |sum, &(x, w)| Ok(sum + w * f(x)?))
}

/// The one log-gamma ratio `ln Gamma(base+shift) - ln Gamma(base)` of scalar laws,
/// keeping a shift below one ulp; distant shifts difference the log gammas.
pub(super) fn log_gamma_ratio(base: f64, shift: f64) -> Result<f64, SciOxideError> {
    if shift.abs() <= base * 0.125 {
        finite(shift * average(|t| scioxide_special::digamma(base + shift * t))?)
    } else {
        finite(scioxide_special::ln_gamma(base + shift)? - scioxide_special::ln_gamma(base)?)
    }
}

/// `ln Gamma(z+shift) - ln Gamma(z) - shift ln z`: the log-gamma ratio with its
/// leading term removed analytically, from Stirling's series once `z >= 32`.
pub(super) fn log_gamma_ratio_remainder(z: f64, shift: f64) -> Result<f64, SciOxideError> {
    if z < 32.0 {
        return Ok(scioxide_special::ln_gamma(z + shift)?
            - scioxide_special::ln_gamma(z)?
            - shift * z.ln());
    }
    let t = shift / z;
    let mut power = t * t;
    let mut residual = 0.0;
    for n in 2..=12 {
        residual += if n % 2 == 0 {
            -power / n as f64
        } else {
            power / n as f64
        };
        power *= t;
    }
    Ok(z * residual
        + (shift - 0.5) * t.ln_1p()
        + crate::distribution::discrete_math::factorial_stirling_correction(z + shift)
        - crate::distribution::discrete_math::factorial_stirling_correction(z))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn narrow_normal_intervals_keep_the_density_limit() {
        let width = 1e-12;
        let result = normal_log_interval(2., 2. + width).unwrap();
        let expected = normal_log_density(2.) + (2. + width - 2.).ln();
        assert!((result - expected).abs() < 1e-10);
        assert_eq!(normal_log_interval(1., 1.).unwrap(), f64::NEG_INFINITY);
        assert!(normal_log_interval(2., 1.).is_err());
    }

    #[test]
    fn log_gamma_ratio_forms_agree() {
        for (z, shift) in [
            (0.75, 0.05),
            (3.0, -0.25),
            (40.0, 0.5),
            (1.0e6, -0.5),
            (2.5, 7.0),
        ] {
            let (high, low) = (
                scioxide_special::ln_gamma(z + shift).unwrap(),
                scioxide_special::ln_gamma(z).unwrap(),
            );
            // The direct difference is only as accurate as its cancelled terms.
            let ratio = log_gamma_ratio(z, shift).unwrap();
            let bound = 4.0 * f64::EPSILON * (high.abs() + low.abs()) + 1e-13 * ratio.abs();
            assert!((ratio - (high - low)).abs() <= bound, "{z} {shift}");
            let remainder = log_gamma_ratio_remainder(z, shift).unwrap();
            assert!((shift * z.ln() + remainder - ratio).abs() <= 1e-12 * (1.0 + ratio.abs()));
        }
        // A sub-ULP shift keeps its first-order effect.
        let tiny = log_gamma_ratio(5.0, 1e-20).unwrap();
        assert!((tiny / 1e-20 - scioxide_special::digamma(5.0).unwrap()).abs() < 1e-14);
    }

    #[test]
    fn beta_leading_term_applies_below_rounding() {
        let (a, b) = (2.5_f64, 4.0);
        let log_beta = || scioxide_special::ln_beta(a, b);
        let tiny = beta_leading_log(a, b, -60.0, log_beta).unwrap().unwrap();
        let exact = scioxide_special::beta_inc_log(a, b, (-60.0_f64).exp()).unwrap();
        assert!((tiny - exact).abs() <= 1e-13 * exact.abs());
        assert!(beta_leading_log(a, b, -20.0, log_beta).unwrap().is_none());
        // It stays finite where x itself underflows.
        assert!(
            beta_leading_log(a, b, -1.0e4, log_beta)
                .unwrap()
                .unwrap()
                .is_finite()
        );
    }
}
