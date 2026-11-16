//! Region-specific `f64` gamma kernels; f32 callers round once.
// Coefficients: Boost.Math 1.89 lgamma_small (John Maddock 2006,
// Matt Borland 2024), Boost Software License 1.0; Cephes gamma rational
// (Stephen L. Moshier, 1984-2000). See THIRD_PARTY_NOTICES.md.
#![allow(clippy::excessive_precision)]
use super::gamma_series::{
    gamma_recurrence_wide, reciprocal_gamma_minus_one, stirling_f64, stirling_wide,
};
use crate::support::polynomial::{PolynomialPolicy, polynomial};
use crate::support::wide::{Wide, ln as ln_wide, two_product};
use scioxide_core::{RealScalar, SciOxideError};

use super::constants::LOG_SQRT_TWO_PI;

/// Direct two-term gamma for cancellation-sensitive recurrence ratios.
/// Its recurrence product stays finite for 1e-150 <= x < 64.
pub(crate) fn gamma_positive_wide(x: Wide) -> Wide {
    gamma_positive_coefficients(x, &crate::support::reciprocal_gamma::RECIPROCAL_GAMMA[1..])
}

fn gamma_positive_coefficients(x: Wide, coefficients: &[(f64, f64)]) -> Wide {
    let (x, product) = gamma_recurrence_wide(x);
    let offset = if x.hi < 0.5 { x } else { x - Wide::from(1.0) };
    let mut reciprocal = Wide::from(1.0) + reciprocal_gamma_minus_one(offset, coefficients);
    if x.hi < 0.5 {
        reciprocal = reciprocal * x;
    }
    product / reciprocal
}

pub(crate) fn beta_moderate(a: Wide, b: Wide) -> Wide {
    gamma_ratio_factor(a) * gamma_ratio_factor(b) / gamma_ratio_factor(a + b)
}

/// Mixed-precision ratio and a relative uncertainty for final rounding.
pub(crate) fn beta_moderate_fast(a: Wide, b: Wide) -> (Wide, f64) {
    let (ga, ea) = gamma_positive_fast(a);
    let (gb, eb) = gamma_positive_fast(b);
    let (gs, es) = gamma_positive_fast(a + b);
    (ga * gb / gs, ea + eb + es)
}

/// Gamma and relative uncertainty for 1e-150 <= x < 128.
/// The scalar polynomial tail starts at u^5; only four leading coefficients
/// keep compensation. Unit decrements in this recurrence are exact.
pub(crate) fn gamma_positive_fast(mut x: Wide) -> (Wide, f64) {
    let mut product = Wide::from(1.0);
    while x.hi > 1.5 {
        x.hi -= 1.0;
        product = super::wide::product(product, x);
    }
    let offset = if x.hi < 0.5 { x } else { x - Wide::from(1.0) };
    let coefficients = &super::reciprocal_gamma_short::RECIPROCAL_GAMMA_SHORT;
    let (tail, magnitude) =
        coefficients[4..]
            .iter()
            .rev()
            .fold((0.0_f64, 0.0_f64), |(tail, magnitude), &(hi, _)| {
                (
                    tail * offset.hi + hi,
                    magnitude * offset.hi.abs() + hi.abs(),
                )
            });
    let mut reciprocal =
        Wide::from(1.0) + offset * super::polynomial::compensated(offset, &coefficients[..4], tail);
    // Sixteen scalar Horner stages, including coefficient rounding, fit
    // within 64 epsilon times the absolute polynomial. The table truncation
    // is below 2.4e-25; retain margin for the compensated head and recurrence.
    let uncertainty = (64.0 * f64::EPSILON * offset.hi.abs().powi(5) * magnitude + 3.0e-25)
        / reciprocal.hi.abs()
        + 512.0 * f64::EPSILON * f64::EPSILON;
    if x.hi < 0.5 {
        reciprocal = reciprocal * x;
    }
    (product / reciprocal, uncertainty)
}

/// Reciprocal-gamma Chebyshev interpolant on [-1/2,1/2], degree 20.
/// Independently generated at 240 digits; absolute truncation < 2.4e-25.
/// Log-beta switches to the long series before cancellation amplifies it.
fn gamma_ratio_factor(x: Wide) -> Wide {
    gamma_positive_coefficients(
        x,
        &crate::support::reciprocal_gamma_short::RECIPROCAL_GAMMA_SHORT,
    )
}

/// Uncombined terms keep their low components when log-beta cancels.
fn ln_positive_parts(x: Wide) -> [Wide; 3] {
    let (x, product) = gamma_recurrence_wide(x);
    let offset = if x.hi < 0.5 { x } else { x - Wide::from(1.0) };
    [
        -ln_wide(
            Wide::from(1.0)
                + reciprocal_gamma_minus_one(
                    offset,
                    &super::reciprocal_gamma::RECIPROCAL_GAMMA[1..],
                ),
        ),
        ln_wide(product),
        if x.hi < 0.5 { -ln_wide(x) } else { Wide::ZERO },
    ]
}

pub(crate) fn ln_positive_wide(x: Wide) -> Wide {
    if x.hi < 32.0 {
        return ln_positive_parts(x)
            .into_iter()
            .fold(Wide::ZERO, |sum, term| sum + term);
    }
    (x - Wide::from(0.5)) * ln_wide(x) - x + LOG_SQRT_TWO_PI + stirling_wide(x)
}

pub(crate) fn ln_negative(x: f64) -> f64 {
    if x <= -8.0 {
        let reflected = ln_positive(1.0 - x);
        let log_sine = crate::util::sin_pi(x).abs().ln();
        let value = (core::f64::consts::PI.ln() - log_sine) - reflected;
        // When the result retains at least two thirds of the component
        // magnitude, reflection does not amplify their rounding envelope.
        if 1.5 * value.abs() >= reflected.abs() + log_sine.abs() + core::f64::consts::PI.ln() {
            return value;
        }
    }
    ln_negative_wide(Wide::from(x)).value()
}

/// Keep the low logarithmic component for combined signed gamma ratios.
pub(crate) fn ln_negative_wide(mut x: Wide) -> Wide {
    // On moderate negative arguments recurrence avoids trigonometric error
    // amplification at the zeros of log|Gamma| altogether.
    if x.hi > -32.0 {
        let mut argument = x;
        let mut product = Wide::from(1.0);
        while argument.hi < 0.0 {
            product = product * (-argument);
            argument = argument + Wide::from(1.0);
        }
        let reciprocal = if argument.hi < 0.5 {
            argument
                * (Wide::from(1.0)
                    + reciprocal_gamma_minus_one(
                        argument,
                        &super::reciprocal_gamma::RECIPROCAL_GAMMA[1..],
                    ))
        } else {
            Wide::from(1.0)
                + reciprocal_gamma_minus_one(
                    argument - Wide::from(1.0),
                    &super::reciprocal_gamma::RECIPROCAL_GAMMA[1..],
                )
        };
        return -((product * reciprocal) - Wide::from(1.0)).ln_1p();
    }
    x = -x;
    let distance = x - Wide::from(x.hi.round());
    let phase = crate::support::constants::PI
        * if distance.hi < 0.0 {
            -distance
        } else {
            distance
        };
    let square = phase * phase;
    let mut term = phase;
    let mut sine = term;
    for index in 1..20 {
        term = -(term * square).div_f64(f64::from(2 * index * (2 * index + 1)));
        sine = sine + term;
    }
    crate::support::constants::PI.ln() - sine.ln() - ln_positive_wide(x + Wide::from(1.0))
}

pub(crate) const G: f64 = 6.024680040776729583740234375;
const P0: [f64; 7] = [
    -0.180355685678449379109e-1,
    0.25126649619989678683e-1,
    0.494103151567532234274e-1,
    0.172491608709613993966e-1,
    -0.259453563205438108893e-3,
    -0.541009869215204396339e-3,
    -0.324588649825948492091e-4,
];
const Q0: [f64; 8] = [
    0.1e1,
    0.196202987197795200688e1,
    0.148019669424231326694e1,
    0.541391432071720958364e0,
    0.988504251128010129477e-1,
    0.82130967464889339326e-2,
    0.224936291922115757597e-3,
    -0.223352763208617092964e-6,
];
const P1: [f64; 7] = [
    0.490622454069039543534e-1,
    -0.969117530159521214579e-1,
    -0.414983358359495381969e0,
    -0.406567124211938417342e0,
    -0.158413586390692192217e0,
    -0.240149820648571559892e-1,
    -0.100346687696279557415e-2,
];
const Q1: [f64; 7] = [
    0.1e1,
    0.302349829846463038743e1,
    0.348739585360723852576e1,
    0.191415588274426679201e1,
    0.507137738614363510846e0,
    0.577039722690451849648e-1,
    0.195768102601107189171e-2,
];
const P2: [f64; 6] = [
    -0.292329721830270012337e-1,
    0.144216267757192309184e0,
    -0.142440390738631274135e0,
    0.542809694055053558157e-1,
    -0.850535976868336437746e-2,
    0.431171342679297331241e-3,
];
const Q2: [f64; 7] = [
    0.1e1,
    -0.150169356054485044494e1,
    0.846973248876495016101e0,
    -0.220095151814995745555e0,
    0.25582797155975869989e-1,
    -0.100666795539143372762e-2,
    -0.827193521891290553639e-6,
];
const NUM: [f64; 13] = [
    23531376880.41075968857200767445163675473,
    42919803642.64909876895789904700198885093,
    35711959237.35566804944018545154716670596,
    17921034426.03720969991975575445893111267,
    6039542586.35202800506429164430729792107,
    1439720407.311721673663223072794912393972,
    248874557.8620541565114603864132294232163,
    31426415.58540019438061423162831820536287,
    2876370.628935372441225409051620849613599,
    186056.2653952234950402949897160456992822,
    8071.672002365816210638002902272250613822,
    210.8242777515793458725097339207133627117,
    2.506628274631000270164908177133837338626,
];
const DEN: [f64; 13] = [
    0.0,
    39916800.0,
    120543840.0,
    150917976.0,
    105258076.0,
    45995730.0,
    13339535.0,
    2637558.0,
    357423.0,
    32670.0,
    1925.0,
    66.0,
    1.0,
];
const GAMMA_P: [f64; 7] = [
    9.99999999999999996796E-1,
    4.94214826801497100753E-1,
    2.07448227648435975150E-1,
    4.76367800457137231464E-2,
    1.04213797561761569935E-2,
    1.19135147006586384913E-3,
    1.60119522476751861407E-4,
];
const GAMMA_Q: [f64; 8] = [
    1.00000000000000000320E0,
    7.14304917030273074085E-2,
    -2.34591795718243348568E-1,
    3.58236398605498653373E-2,
    1.18139785222060435552E-2,
    -4.45641913851797240494E-3,
    5.39605580493303397842E-4,
    -2.31581873324120129819E-5,
];
pub(crate) const PARTIAL: [f64; 13] = [
    0.999999999999999907345313440276,
    235.180704707206475554363871623,
    -354.2708755572972462948932243,
    157.916827458998601140226132233,
    -21.2292982783458128811780076995,
    0.509522237950569156589026752027,
    -0.000161418930719326622799653958277,
    -0.00000293126204719267739602633078057,
    0.00000327483970129359111869811954511,
    -0.00000205836001103943463589392532714,
    0.000000925399062328335512069471627674,
    -0.000000266144214700422721856918505975,
    0.0000000361457403131915465724832869354,
];

pub(crate) fn lanczos(x: f64) -> f64 {
    if x <= 1.0 {
        polynomial(x, &NUM, PolynomialPolicy::Fused) / polynomial(x, &DEN, PolynomialPolicy::Fused)
    } else {
        let reciprocal = 1.0 / x;
        let numerator = polynomial(reciprocal, &NUM, PolynomialPolicy::FusedDescending);
        let denominator = polynomial(reciprocal, &DEN, PolynomialPolicy::FusedDescending);
        numerator / denominator
    }
}

/// Common power factor x^a exp(-x)/Gamma(a), combining the Lanczos
/// exponent before projection so central tails do not subtract large logs.
pub(crate) fn gamma_prefix(a: f64, x: f64) -> f64 {
    let t = Wide::from(a) + Wide::from(G - 0.5);
    let ratio = Wide::from(x) / t;
    let log_ratio = if ratio.hi < f64::MIN_POSITIVE {
        // A subnormal ratio loses relative precision even when x is normal.
        // Preserve its logarithm until the final projection.
        ln_wide(Wide::from(x)) - ln_wide(t)
    } else {
        let logarithm = ratio.hi.ln();
        // Reserve half the 32-ULP tail envelope for logarithm rounding and
        // its amplification by a. Large powers retain the low log component.
        if a <= 32.0 && a * logarithm.abs() <= 16.0 {
            Wide::from(logarithm) + Wide::from(ratio.lo / ratio.hi)
        } else {
            ln_wide(ratio)
        }
    };
    let power = log_ratio.mul_f64(a);
    let logarithm = power + (t - Wide::from(x)) + Wide::from(0.5 * t.hi.ln() - lanczos(a).ln());
    logarithm.exp()
}

pub(crate) fn ln_positive(mut x: f64) -> f64 {
    if x >= 16.0 {
        let inverse = 1.0 / x;
        let square = inverse * inverse;
        return scioxide_core::fast_mul_add(
            x - 0.5,
            x.ln() - 1.0,
            0.41893853320467274178032973640562,
        ) + stirling_f64(inverse, square);
    }
    if x < 1.0 {
        if x < 1.0e-16 {
            return -x.ln();
        }
        if x >= 0.5 {
            // Preserve x-1 when x+1 rounds to exactly 2. The shifted
            // approximation is evaluated with the original displacement.
            let e = x - 1.0;
            let factor = x * e;
            return scioxide_core::fast_mul_add(
                factor,
                polynomial(-e, &P2, PolynomialPolicy::Fused)
                    / polynomial(-e, &Q2, PolynomialPolicy::Fused),
                factor * 0.452017307281494140625,
            ) - e.ln_1p();
        }
        return ln_positive(x + 1.0) - x.ln();
    }
    let mut recurrence = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        recurrence *= x;
    }
    let value = if x == 1.0 || x == 2.0 {
        0.0
    } else if x > 2.0 {
        let e = x - 2.0;
        let factor = e * (x + 1.0);
        scioxide_core::fast_mul_add(
            factor,
            polynomial(e, &P0, PolynomialPolicy::Fused)
                / polynomial(e, &Q0, PolynomialPolicy::Fused),
            factor * 0.158963680267333984375,
        )
    } else if x <= 1.5 {
        let e = x - 1.0;
        let factor = e * (x - 2.0);
        scioxide_core::fast_mul_add(
            factor,
            polynomial(e, &P1, PolynomialPolicy::Fused)
                / polynomial(e, &Q1, PolynomialPolicy::Fused),
            factor * 0.52815341949462890625,
        )
    } else {
        let e = 2.0 - x;
        let factor = (x - 1.0) * (x - 2.0);
        scioxide_core::fast_mul_add(
            factor,
            polynomial(e, &P2, PolynomialPolicy::Fused)
                / polynomial(e, &Q2, PolynomialPolicy::Fused),
            factor * 0.452017307281494140625,
        )
    };
    value + recurrence.ln()
}

#[inline]
fn normalized(value: f64) -> Result<(f64, i32), SciOxideError> {
    // The bounded gamma powers and exponentials are finite and positive.
    // Core's IEEE decomposition gives their exact [0.5,1) frame directly.
    if value == 0.0 || !value.is_finite() {
        crate::support::check_finite(value)?;
        return Ok((value, 1));
    }
    let (significand, exponent) = crate::support::wide::finite_parts(value);
    Ok((significand * 0.5, exponent + 1))
}

fn scaled_ratio(numerator: &[f64], denominator: &[f64], exponent: i32, correction: f64) -> f64 {
    // The leading factor is exact; only later factors need product residuals.
    let product = |factors: &[f64]| {
        factors
            .split_first()
            .map_or(Wide::from(1.0), |(&first, rest)| {
                rest.iter()
                    .fold(Wide::from(first), |value, &factor| value.mul_f64(factor))
            })
    };
    let Wide { hi: nh, lo: nl } = product(numerator);
    let Wide { hi: dh, lo: dl } = product(denominator);
    let quotient = nh / dh;
    let back = two_product(quotient, dh);
    let residual = ((nh - back.hi) - back.lo) + nl - quotient * dl;
    let low = scioxide_core::fast_mul_add(quotient, correction, residual / dh);
    (quotient + low).scalbn(exponent)
}

/// Positive gamma with recurrence below 16 and a scaled Stirling product above.
/// The Stirling and square-root constant corrections enter the quotient's low
/// word, keeping their small contributions through the final rounding.
pub(crate) fn gamma_positive(mut x: f64) -> Result<f64, SciOxideError> {
    if x < 0.5 {
        return Ok(gamma_positive(x + 1.0)? / x);
    }
    if x < 16.0 {
        let mut factor = 1.0;
        while x > 2.0 {
            x -= 1.0;
            factor *= x;
        }
        if x < 1.0 {
            factor /= x;
            x += 1.0;
        }
        if x == 1.0 || x == 2.0 {
            return Ok(factor);
        }
        let e = x - 1.0;
        return Ok(factor
            * (polynomial(e, &GAMMA_P, PolynomialPolicy::Fused)
                / polynomial(e, &GAMMA_Q, PolynomialPolicy::Fused))
            / x);
    }
    if x > 171.625 {
        return Ok(f64::INFINITY);
    }
    // exp(Stirling correction)-1 through degree 18; at x >= 16 the
    // first omitted term is below 2e-23. Independent grouped chains keep
    // the full polynomial without a serial eighteen-stage Horner path.
    const CORRECTION: [f64; 18] = [
        0.08333333333333333,
        0.003472222222222222,
        -0.0026813271604938273,
        -0.00022947209362139917,
        0.0007840392217200666,
        6.972813758365857e-5,
        -0.0005921664373536939,
        -5.171790908260592e-5,
        0.0008394987206720873,
        7.204895416020011e-5,
        -0.0019144384985654776,
        -0.00016251626278391583,
        0.00640336283380807,
        0.0005401647678926045,
        -0.02952788094569912,
        -0.0024817436002649977,
        0.17954011706123485,
        0.015056113040026424,
    ];
    let inverse = 1.0 / x;
    let correction = inverse * polynomial(inverse, &CORRECTION, PolynomialPolicy::Grouped);
    let (power, power_exponent) = normalized(x.powf(x * 0.5 - 0.25))?;
    let (exponential, exponential_exponent) = normalized(x.exp())?;
    let sqrt_two_pi = 2.5066282746310007;
    let constant_correction = -1.8328579980459167e-16 / sqrt_two_pi;
    // Apply (1 + correction)(1 + constant_correction) - 1 in the
    // quotient's low word. Keeping one out of the numerator preserves
    // the small correction and removes one compensated product. Retain
    // the cross term, which reaches 3.9e-19 at the x = 16 handoff.
    let correction = scioxide_core::fast_mul_add(
        correction,
        constant_correction,
        correction + constant_correction,
    );
    Ok(scaled_ratio(
        &[sqrt_two_pi, power, power],
        &[exponential],
        2 * power_exponent - exponential_exponent,
        correction,
    ))
}

pub(crate) fn gamma_negative(x: f64) -> Result<f64, SciOxideError> {
    // Gamma(x) = -pi / (x Gamma(-x) sin(pi x)); inverse factors
    // retain subnormal values after Gamma(-x) itself would overflow.
    let y = -x;
    if y < 0.5 {
        // Recurrence avoids an underflowing y*sin(pi*x) and an overflowing
        // Gamma(y) when x approaches the pole at zero.
        return Ok(gamma_positive(1.0 - y)? / x);
    }
    let sine = crate::util::sin_pi(x);
    if y < 16.0 {
        return Ok(scaled_ratio(
            &[core::f64::consts::PI],
            &[y, sine, gamma_positive(y)?],
            0,
            0.0,
        ));
    }
    if y > 200.0 {
        return Ok(if sine < 0.0 { -0.0 } else { 0.0 });
    }
    let t = y + (G - 0.5);
    let z = t - y;
    let tail = (y - (t - z)) + ((G - 0.5) - z);
    let (power, power_exponent) = normalized(t.powf(y * 0.5 - 0.25))?;
    let (exponential, exponential_exponent) = normalized(t.exp())?;
    Ok(scaled_ratio(
        &[core::f64::consts::PI, exponential],
        &[y, sine, lanczos(y), power, power],
        exponential_exponent - 2 * power_exponent,
        G * tail / t,
    ))
}

/// Digamma and the magnitude of the terms it sums. On the negative axis the
/// reflection `ψ(1−x) − π cot(πx)` cancels near each zero, so its rounding is
/// absolute at the scale `|ψ(1−x)| + |π cot(πx)|` rather than relative to `|ψ|`.
pub(crate) fn digamma_scaled(x: f64) -> (f64, f64) {
    if x < 0.0 {
        // 1 − x may round; restore its residual with the asymptotic trigamma
        // estimate, whose few-percent error is negligible on s > 1.
        let shifted = Wide::from(1.0) - Wide::from(x);
        let s = shifted.hi;
        let trigamma = (1.0 + (0.5 + 1.0 / (6.0 * s)) / s) / s;
        let reflected = digamma(s) + shifted.lo * trigamma;
        let cotangent = pi_cot_pi(x);
        (reflected - cotangent, reflected.abs() + cotangent.abs())
    } else {
        let value = digamma(x);
        (value, value.abs())
    }
}

/// `π cot(πx)` after the exact reduction `r = x − round(x)`. Beside
/// half-integers the complementary `tan(π(1/2 − |r|))`, with an exact
/// argument, keeps the small cotangent's relative accuracy.
fn pi_cot_pi(x: f64) -> f64 {
    use core::f64::consts::PI;
    let reduced = x - x.round();
    if reduced.abs() <= 0.25 {
        PI / (PI * reduced).tan()
    } else {
        (PI * (PI * (0.5 - reduced.abs())).tan()).copysign(reduced)
    }
}

pub(crate) fn digamma(mut x: f64) -> f64 {
    if x < 0.0 {
        return digamma_scaled(x).0;
    }
    if x >= 16.0 {
        let inverse = 1.0 / x;
        let bernoulli = [
            1.0 / 12.0,
            -1.0 / 120.0,
            1.0 / 252.0,
            -1.0 / 240.0,
            1.0 / 132.0,
            -691.0 / 32760.0,
            1.0 / 12.0,
            -3617.0 / 8160.0,
        ];
        return x.ln()
            - 0.5 * inverse
            - inverse
                * inverse
                * polynomial(inverse * inverse, &bernoulli, PolynomialPolicy::Fused);
    }
    let mut correction = 0.0;
    while x > 2.0 {
        x -= 1.0;
        correction += 1.0 / x;
    }
    if x < 1.0 {
        correction -= 1.0 / x;
        x += 1.0;
    }
    let p = [
        0.25479851061131551,
        -0.32555031186804491,
        -0.65031853770896507,
        -0.28919126444774784,
        -0.045251321448739056,
        -0.0020713321167745952,
    ];
    let q = [
        1.0,
        2.0767117023730469,
        1.4606242909763515,
        0.43593529692665969,
        0.054151797245674225,
        0.0021284987017821144,
        -0.55789841321675513e-6,
    ];
    let displacement = ((x - 1569415565.0 / 1073741824.0)
        - (381566830.0 / 1073741824.0) / 1073741824.0)
        - 0.9016312093258695918615325266959189453125e-19;
    scioxide_core::fast_mul_add(
        displacement,
        polynomial(x - 1.0, &p, PolynomialPolicy::Fused)
            / polynomial(x - 1.0, &q, PolynomialPolicy::Fused)
            + 0.99558162689208984,
        correction,
    )
}
