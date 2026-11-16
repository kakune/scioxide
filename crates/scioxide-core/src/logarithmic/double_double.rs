//! Double-double arithmetic for the certified positive reduction.
//!
//! With u=2^-53, normalized addition has absolute error at most
//! 4*u²*(|a|+|b|), and FMA multiplication at most 8*u²*|a*b|, apart
//! from gradual underflow. The high product and its FMA residual are exact;
//! only operations on low components round. TwoSum renormalizes each result.
//!
//! The first ln(2) part has 32 significant bits: multiplication by an integer
//! of magnitude <=1100 is exact. The next two parts leave <2^-140 absolute
//! constant error (<2^-129 after multiplication). Subtracting the high part
//! satisfies Sterbenz except at k=0, where it is the identity. The remaining
//! reduction error is bounded by 16*u² in absolute units. Rounded exponent
//! selection gives |r|<0.022 (including ln(2)/32's reduction-boundary slack).
//! The degree-16 expm1 remainder, after four doublings, is <2^-137.
//! Reciprocal-factorial errors are <u² relative. Horner's low-component
//! errors form a geometric series with ratio <1/32; their contribution after
//! multiplication by r² and four doublings is <2*u². The initial r addition
//! and four c*(2+c) doublings contribute <32*u² to relative exponential error.
//! Including reduction and final unit addition gives eta_exp=64*u²=2^-100.
//! Scaling can additionally lose at most one minimum subnormal per component.
//!
//! The logarithmic Newton step has a seed error delta bounded by eight ULPs
//! of ln_1p(r_hi), including r_lo's effect. Its remaining analytic error is
//! <=delta² (|delta|<2^-40 here). Exponential, division and addition errors
//! are bounded by eta_log=2^-98*(1+|log|)+delta². This is an absolute bound,
//! separate from the accumulated exponential and extraction errors.

use crate::arithmetic::two_sum;
use crate::{RangeAccumulator, RealScalar};

use super::expansion::Expansion;

#[derive(Clone, Copy, Debug)]
pub(super) struct DoubleDouble {
    pub(super) hi: f64,
    pub(super) lo: f64,
}

// MPFR 512-bit successive nearest-even residuals. The high ln(2) part is
// deliberately truncated so the Cody-Waite integer product is exact for |k|<=1732.
pub(super) const LN_2: [f64; 3] = [
    0.693_147_180_369_123_8,
    1.908_214_929_270_587_7e-10,
    1.161_222_722_936_253_2e-26,
];

pub(super) const RECIPROCAL_FACTORIALS: [DoubleDouble; 15] = [
    DoubleDouble { hi: 0.5, lo: 0.0 },
    DoubleDouble {
        hi: 0.166_666_666_666_666_66,
        lo: 9.251_858_538_542_97e-18,
    },
    DoubleDouble {
        hi: 0.041_666_666_666_666_664,
        lo: 2.312_964_634_635_742_7e-18,
    },
    DoubleDouble {
        hi: 0.008_333_333_333_333_333,
        lo: 1.156_482_317_317_871_4e-19,
    },
    DoubleDouble {
        hi: 0.001_388_888_888_888_889,
        lo: -5.300_543_954_373_577e-20,
    },
    DoubleDouble {
        hi: 0.000_198_412_698_412_698_4,
        lo: 1.720_955_829_342_070_5e-22,
    },
    DoubleDouble {
        hi: 2.480_158_730_158_73e-5,
        lo: 2.151_194_786_677_588_2e-23,
    },
    DoubleDouble {
        hi: 2.755_731_922_398_589_3e-6,
        lo: -1.858_393_274_046_472e-22,
    },
    DoubleDouble {
        hi: 2.755_731_922_398_589e-7,
        lo: 2.376_771_462_225_029_7e-23,
    },
    DoubleDouble {
        hi: 2.505_210_838_544_172e-8,
        lo: -1.448_814_070_935_912e-24,
    },
    DoubleDouble {
        hi: 2.087_675_698_786_81e-9,
        lo: -1.207_345_059_113_26e-25,
    },
    DoubleDouble {
        hi: 1.605_904_383_682_161_3e-10,
        lo: 1.258_529_458_875_209_8e-26,
    },
    DoubleDouble {
        hi: 1.147_074_559_772_972_5e-11,
        lo: 2.065_551_275_283_074_5e-28,
    },
    DoubleDouble {
        hi: 7.647_163_731_819_816e-13,
        lo: 7.038_728_777_334_53e-30,
    },
    DoubleDouble {
        hi: 4.779_477_332_387_385e-14,
        lo: 4.399_205_485_834_081e-31,
    },
];

impl DoubleDouble {
    pub(super) const fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }

    pub(super) fn pair(hi: f64, lo: f64) -> Self {
        let (hi, lo) = two_sum(hi, lo);
        Self { hi, lo }
    }

    pub(super) fn add(self, rhs: Self) -> Self {
        let (hi, mut lo) = two_sum(self.hi, rhs.hi);
        let (tail, residual) = two_sum(self.lo, rhs.lo);
        lo += tail;
        let (hi, lo) = two_sum(hi, lo);
        Self::pair(hi, lo + residual)
    }

    pub(super) fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    pub(super) fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    pub(super) fn mul(self, rhs: Self) -> Self {
        let hi = self.hi * rhs.hi;
        let lo = RealScalar::mul_add(self.hi, rhs.hi, -hi);
        let lo = RealScalar::mul_add(self.hi, rhs.lo, lo);
        let lo = RealScalar::mul_add(self.lo, rhs.hi, lo);
        let lo = RealScalar::mul_add(self.lo, rhs.lo, lo);
        Self::pair(hi, lo)
    }

    fn div(self, rhs: Self) -> Self {
        let q = Self::from(self.hi / rhs.hi);
        let residual = self.sub(rhs.mul(q));
        q.add(Self::from((residual.hi + residual.lo) / rhs.hi))
    }

    fn scale(self, exponent: i32) -> Self {
        Self::pair(
            RealScalar::scalbn(self.hi, exponent),
            RealScalar::scalbn(self.lo, exponent),
        )
    }

    fn reduced_expm1(self) -> (Self, i32) {
        if self.hi == 0.0 && self.lo == 0.0 {
            return (Self::from(0.0), 0);
        }
        let k = RealScalar::round(self.hi * core::f64::consts::LOG2_E) as i32;
        let kf = f64::from(k);
        let high = Self::from(kf * LN_2[0]);
        let middle = Self::from(kf).mul(Self::from(LN_2[1]));
        let low = Self::from(kf).mul(Self::from(LN_2[2]));
        let r = self.sub(high).sub(middle).sub(low).scale(-4);
        let polynomial = RECIPROCAL_FACTORIALS[..14]
            .iter()
            .rev()
            .fold(RECIPROCAL_FACTORIALS[14], |p, &coefficient| {
                p.mul(r).add(coefficient)
            });
        let mut correction = r.add(r.mul(r).mul(polynomial));
        for _ in 0..4 {
            correction = correction.mul(Self::from(2.0).add(correction));
        }
        (correction, k)
    }

    // Positive-reduction shifts are in [-740, 0]; logarithmic seeds are in
    // [0, ln(usize::MAX)], so all reduction products satisfy |k|<=1100.
    pub(super) fn exp(self) -> Self {
        let (mantissa, exponent) = self.exp_scaled();
        mantissa.scale(exponent)
    }

    // Retain the exponent until the reduced exponential is reconstructed.
    fn exp_scaled(self) -> (Self, i32) {
        let (correction, k) = self.reduced_expm1();
        (Self::from(1.0).add(correction), k)
    }

    fn expm1(self) -> Self {
        let (correction, k) = self.reduced_expm1();
        if k == 0 {
            correction
        } else {
            Self::from(1.0)
                .add(correction)
                .scale(k)
                .sub(Self::from(1.0))
        }
    }

    pub(super) fn ln_1p(self) -> (Self, f64) {
        let seed = RealScalar::ln_1p(self.hi);
        let value = Self::from(seed).expm1();
        let correction = self.sub(value).div(Self::from(1.0).add(value));
        let log = Self::from(seed).add(correction);
        let delta = 8.0 * super::destination_ulp::<f64>(seed);
        let error = super::add_up(
            super::mul_up(
                RealScalar::scalbn(1.0_f64, -98),
                super::add_up(1.0, log.hi.abs()),
            ),
            super::mul_up(delta, delta),
        );
        (log, error)
    }

    // The exact bin extraction truncates two 53-bit chunks. Its omitted tail
    // is <2^-105 of the leading binary frame, hence <2^-105 of a positive sum.
    pub(super) fn from_sum(sum: &RangeAccumulator<f64>) -> Self {
        let (parts, exponent) = sum.expansion();
        Self::pair(parts[0], parts[1]).scale(exponent)
    }

    pub(super) fn expansion(self) -> Expansion {
        Expansion::from(self.hi).add(Expansion::from(self.lo))
    }
}
