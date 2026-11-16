//! Starting values of indexed zeros from DLMF 7.13, 9.9.18–19, 10.21 and 10.70.
use super::{KelvinZeroFunction as K, SpecialZeroFamily as F, evaluation::C, starts};
const PI: f64 = core::f64::consts::PI;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar;

/// DLMF 10.70.1–2 through `t⁻⁴`. Derivatives are rotated order-one
/// functions (10.63.3–4), so they use `μ = 4` and their own phase offsets.
pub(super) fn kelvin(function: K, index: usize) -> f64 {
    let (values, offset, mu, second) = match function {
        K::Ber => (&starts::BER, -0.375, 0.0, false),
        K::Bei => (&starts::BEI, 0.125, 0.0, false),
        K::Ker => (&starts::KER, -0.625, 0.0, true),
        K::Kei => (&starts::KEI, -0.125, 0.0, true),
        K::BerPrime => (&starts::BERP, 0.375, 4.0, false),
        K::BeiPrime => (&starts::BEIP, -0.125, 4.0, false),
        K::KerPrime => (&starts::KERP, -0.375, 4.0, true),
        K::KeiPrime => (&starts::KEIP, 0.125, 4.0, true),
    };
    if let Some(&value) = index.checked_sub(1).and_then(|i| values.get(i)) {
        return value;
    }
    let t = (index as f64 + offset) * PI;
    let q = if second { -1.0 / t } else { 1.0 / t };
    let correction = (mu - 1.0)
        * q
        * (1.0 / 16.0
            + q * (1.0 / 32.0 + q * ((5.0 * mu + 19.0) / 1536.0 + q * 3.0 * (mu - 1.0) / 512.0)));
    core::f64::consts::SQRT_2 * (t + if second { correction } else { -correction })
}

pub(super) fn table(family: F<f64>, index: usize) -> Option<f64> {
    let values = match family {
        F::BesselJ(0.0) => &starts::J0,
        F::BesselJ(1.0) | F::BesselJPrime(0.0) => &starts::J1,
        F::BesselJ(2.5) => &starts::J25,
        F::BesselJPrime(1.0) => &starts::JP1,
        F::BesselY(0.0) => &starts::Y0,
        F::BesselY(1.0) | F::BesselYPrime(0.0) => &starts::Y1,
        F::AiryAi => &starts::AI,
        F::AiryAiPrime => &starts::AIP,
        F::AiryBi => &starts::BI,
        F::AiryBiPrime => &starts::BIP,
        _ => return None,
    };
    index.checked_sub(1).and_then(|i| values.get(i)).copied()
}

pub(super) fn airy(family: F<f64>, index: usize) -> f64 {
    if let Some(value) = table(family, index) {
        return value;
    }
    let prime = matches!(family, F::AiryAiPrime | F::AiryBiPrime);
    let shift = if matches!(family, F::AiryAi | F::AiryBiPrime) {
        0.25
    } else {
        0.75
    };
    let t = 1.5 * core::f64::consts::PI * (index as f64 - shift);
    let q = 1.0 / (t * t);
    let correction = if prime {
        q * (-7.0 / 48.0
            + q * (35.0 / 288.0
                + q * (-181223.0 / 207360.0
                    + q * (18683371.0 / 1244160.0 - q * 91145884361.0 / 191102976.0))))
    } else {
        q * (5.0 / 48.0
            + q * (-5.0 / 36.0
                + q * (77125.0 / 82944.0
                    + q * (-108056875.0 / 6967296.0 + q * 162375596875.0 / 334430208.0))))
    };
    -t.cbrt().powi(2) * (1.0 + correction)
}

/// Returns a start only for large indices (McMahon) or near the turning point (Olver).
pub(super) fn bessel(family: F<f64>, index: usize) -> Option<f64> {
    if !matches!(
        family,
        F::BesselJ(_) | F::BesselY(_) | F::BesselJPrime(_) | F::BesselYPrime(_)
    ) {
        return None;
    }
    if let Some(value) = table(family, index) {
        return Some(value);
    }
    if matches!(family, F::BesselJPrime(0.0)) {
        return bessel(F::BesselJ(1.0), index);
    }
    if matches!(family, F::BesselYPrime(0.0)) {
        return bessel(F::BesselY(1.0), index);
    }
    let (nu, second, prime) = match family {
        F::BesselJ(nu) => (nu, false, false),
        F::BesselY(nu) => (nu, true, false),
        F::BesselJPrime(nu) => (nu, false, true),
        F::BesselYPrime(nu) => (nu, true, true),
        _ => return None,
    };
    let shift = if second == prime { 0.25 } else { 0.75 };
    let beta = core::f64::consts::PI * (index as f64 + nu * 0.5 - shift);
    if beta > 8.0 && beta > 2.0 * (nu * nu + 1.0) {
        let mu = 4.0 * nu * nu;
        let inverse = 1.0 / (8.0 * beta);
        let q = inverse * inverse;
        let correction = if prime {
            inverse
                * (mu
                    + 3.0
                    + q * (4.0 / 3.0 * (7.0 * mu * mu + 82.0 * mu - 9.0)
                        + q * (32.0 / 15.0
                            * (83.0 * mu.powi(3) + 2075.0 * mu * mu - 3039.0 * mu + 3537.0)
                            + q * 64.0 / 105.0
                                * (6949.0 * mu.powi(4) + 296492.0 * mu.powi(3)
                                    - 1248002.0 * mu * mu
                                    + 7414380.0 * mu
                                    - 5853627.0))))
        } else {
            (mu - 1.0)
                * inverse
                * (1.0
                    + q * (4.0 / 3.0 * (7.0 * mu - 31.0)
                        + q * (32.0 / 15.0 * (83.0 * mu * mu - 982.0 * mu + 3779.0)
                            + q * 64.0 / 105.0
                                * (6949.0 * mu.powi(3) - 153855.0 * mu * mu + 1585743.0 * mu
                                    - 6277237.0))))
        };
        return Some(beta - correction);
    }
    if nu >= 32.0 {
        let airy_family = match (second, prime) {
            (false, false) => F::AiryAi,
            (false, true) => F::AiryAiPrime,
            (true, false) => F::AiryBi,
            (true, true) => F::AiryBiPrime,
        };
        let alpha = -airy(airy_family, index) / 2.0_f64.cbrt();
        let root = nu.cbrt();
        if alpha < 0.4 * root * root {
            let a2 = if prime {
                0.3 * alpha * alpha - 0.1 / alpha
            } else {
                0.3 * alpha * alpha
            };
            let a3 = if prime {
                -alpha.powi(3) / 350.0 - 1.0 / 25.0 - 1.0 / (200.0 * alpha.powi(3))
            } else {
                (5.0 - alpha.powi(3)) / 350.0
            };
            return Some(nu + alpha * root + a2 / root + a3 / nu);
        }
    }
    None
}

pub(super) fn complex(f: F<f64>, index: usize) -> C {
    if let Some(values) = match f {
        F::Erf => Some(&starts::ERF),
        F::FresnelC => Some(&starts::FRESNEL_C),
        F::FresnelS => Some(&starts::FRESNEL_S),
        _ => None,
    } && let Some(&[re, im]) = index.checked_sub(1).and_then(|i| values.get(i))
    {
        return C::new(re, im);
    }
    let n = index as f64;
    match f {
        F::Erf => {
            let l = ((n - 0.125) * PI).sqrt();
            let m = (l * (2.0 * PI).sqrt()).ln();
            let a = (1.0 - m + 0.5 * m * m) / (16.0 * l * l * l);
            C::new(l - m / (4.0 * l) + a, l + m / (4.0 * l) + a)
        }
        F::FresnelC | F::FresnelS => {
            let l = (4.0 * n - if matches!(f, F::FresnelC) { 1.0 } else { 0.0 }).sqrt();
            let a = 2.0 / PI * (PI * l).ln();
            C::new(
                l + a * (a * PI - 4.0) / (8.0 * PI * l * l * l),
                a / (2.0 * l),
            )
        }
        F::AiryBi | F::AiryBiPrime => {
            let t = C::new(
                1.5 * PI * (n - if matches!(f, F::AiryBi) { 0.25 } else { 0.75 }),
                0.75 * core::f64::consts::LN_2,
            );
            (t.ln() * (2.0 / 3.0)).exp() * C::new(0.5, 3.0_f64.sqrt() * 0.5)
        }
        F::BesselY(0.0) => C::new(-(n - 0.25) * PI, 0.5 * 3.0_f64.ln()),
        F::BesselY(1.0) | F::BesselYPrime(0.0) => {
            if index == 1 {
                C::new(-0.5, 0.79)
            } else {
                C::new(-(n - 0.75) * PI, 0.5 * 3.0_f64.ln())
            }
        }
        F::BesselYPrime(1.0) => {
            if index == 1 {
                C::new(0.5768, 0.904)
            } else {
                C::new(-(n - 1.25) * PI, 0.5 * 3.0_f64.ln())
            }
        }
        F::FresnelCPrime | F::FresnelSPrime => C::new(
            (2.0 * n
                - if matches!(f, F::FresnelCPrime) {
                    1.0
                } else {
                    0.0
                })
            .sqrt(),
            0.0,
        ),
        _ => C::new(0.0, 0.0),
    }
}
