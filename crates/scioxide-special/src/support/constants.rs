//! Mathematical constants rounded as an `f64` high word and residual.
//!
//! The independent 160/240-digit references live in tests/data/wide_constants.tsv.
use super::wide::Wide;

pub(crate) const PI: Wide = Wide {
    hi: f64::from_bits(0x400921fb54442d18),
    lo: f64::from_bits(0x3ca1a62633145c07),
};
pub(crate) const TAU: Wide = Wide {
    hi: f64::from_bits(0x401921fb54442d18),
    lo: f64::from_bits(0x3cb1a62633145c07),
};
pub(crate) const FRAC_PI_2: Wide = Wide {
    hi: f64::from_bits(0x3ff921fb54442d18),
    lo: f64::from_bits(0x3c91a62633145c07),
};
pub(crate) const FRAC_PI_4: Wide = Wide {
    hi: f64::from_bits(0x3fe921fb54442d18),
    lo: f64::from_bits(0x3c81a62633145c07),
};
pub(crate) const FRAC_1_PI: Wide = Wide {
    hi: f64::from_bits(0x3fd45f306dc9c883),
    lo: f64::from_bits(0xbc76b01ec5417056),
};
pub(crate) const LN_2: Wide = Wide {
    hi: f64::from_bits(0x3fe62e42fefa39ef),
    lo: f64::from_bits(0x3c7abc9e3b39803f),
};
pub(crate) const LN_PI: Wide = Wide {
    hi: f64::from_bits(0x3ff250d048e7a1bd),
    lo: f64::from_bits(0x3c67abf2ad8d5088),
};
pub(crate) const LOG_SQRT_TWO_PI: Wide = Wide {
    hi: f64::from_bits(0x3fed67f1c864beb5),
    lo: f64::from_bits(0xbc865b5a1b7ff5df),
};
pub(crate) const SQRT_PI: Wide = Wide {
    hi: f64::from_bits(0x3ffc5bf891b4ef6b),
    lo: f64::from_bits(0xbc9618f13eb7ca89),
};
pub(crate) const FRAC_2_SQRT_PI: Wide = Wide {
    hi: f64::from_bits(0x3ff20dd750429b6d),
    lo: f64::from_bits(0x3c71ae3a914fed80),
};
pub(crate) const CBRT_2: Wide = Wide {
    hi: f64::from_bits(0x3ff428a2f98d728b),
    lo: f64::from_bits(0xbc7ddc22548ea41e),
};
pub(crate) const TWO_THIRDS: Wide = Wide {
    hi: f64::from_bits(0x3fe5555555555555),
    lo: f64::from_bits(0x3c85555555555555),
};
pub(crate) const EULER_GAMMA: Wide = Wide {
    hi: f64::from_bits(0x3fe2788cfc6fb619),
    lo: f64::from_bits(0xbc56cb90701fbfab),
};
pub(crate) const LN_THREE_QUARTERS: Wide = Wide {
    hi: f64::from_bits(0xbfd269621134db92),
    lo: f64::from_bits(0xbc7e0efadd9db02b),
};
pub(crate) const LN_3: Wide = Wide {
    hi: f64::from_bits(0x3ff193ea7aad030b),
    lo: f64::from_bits(0xbc9a256f99caabeb),
};
pub(crate) const ZETA_2: Wide = Wide {
    hi: f64::from_bits(0x3ffa51a6625307d3),
    lo: f64::from_bits(0x3c81873d8912200c),
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// Decimal integer chunks are exact; the two-word operations keep the
    /// fractional digits that would disappear in a scalar string conversion.
    fn decimal(text: &str) -> Wide {
        let negative = text.starts_with('-');
        let text = text.trim_start_matches('-');
        let (whole, fraction) = text.split_once('.').unwrap();
        let mut value = Wide::from(whole.parse::<f64>().unwrap());
        let mut scale = Wide::from(1.0);
        for chunk in fraction.as_bytes().chunks(9) {
            let factor = 10_u32.pow(chunk.len() as u32);
            scale = scale.div_f64(f64::from(factor));
            let part = core::str::from_utf8(chunk).unwrap().parse::<u32>().unwrap();
            value = value + scale.mul_f64(f64::from(part));
        }
        if negative { -value } else { value }
    }

    #[test]
    fn every_constant_matches_high_precision_digits_and_residual_bits() {
        let constants = [
            ("PI", PI),
            ("TAU", TAU),
            ("FRAC_PI_2", FRAC_PI_2),
            ("FRAC_PI_4", FRAC_PI_4),
            ("FRAC_1_PI", FRAC_1_PI),
            ("LN_2", LN_2),
            ("LN_PI", LN_PI),
            ("LOG_SQRT_TWO_PI", LOG_SQRT_TWO_PI),
            ("SQRT_PI", SQRT_PI),
            ("FRAC_2_SQRT_PI", FRAC_2_SQRT_PI),
            ("CBRT_2", CBRT_2),
            ("TWO_THIRDS", TWO_THIRDS),
            ("EULER_GAMMA", EULER_GAMMA),
            ("LN_THREE_QUARTERS", LN_THREE_QUARTERS),
            ("LN_3", LN_3),
            ("ZETA_2", ZETA_2),
        ];
        let references = include_str!("../../tests/data/wide_constants.tsv");
        let mut count = 0;
        for row in references.lines().filter(|line| !line.starts_with('#')) {
            let mut fields = row.split_whitespace();
            let name = fields.next().unwrap();
            let digits = fields.next().unwrap();
            let high = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let low = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let (_, constant) = constants.iter().find(|(key, _)| *key == name).unwrap();
            assert_eq!(constant.hi.to_bits(), high, "{name} high word");
            assert_eq!(constant.lo.to_bits(), low, "{name} residual");
            let reference = decimal(digits);
            assert!(
                (*constant - reference).value().abs()
                    <= 64.0 * f64::EPSILON * f64::EPSILON * reference.hi.abs(),
                "{name} digits"
            );
            count += 1;
        }
        assert_eq!(count, constants.len());
    }
}
