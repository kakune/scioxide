//! Working-precision positive sums with outward error bounds.
//!
//! The Gal table exponential has error <2*u relative to the exact shifted
//! term. Extreme shifts use owned exp, and logarithms use owned log1p, each
//! bounded by one binary64 ULP. TwoSum keeps the shift residual; correcting
//! the tail exp(hi) by (1+lo) adds rounding errors and O(lo²).
//! For kept shifts in [-745,0], |lo| <= 2^-44, so a conservative term
//! bound is 4*u*term + 4*minimum_subnormal, with u=2^-53.
//!
//! Four independent sums expose exponential and addition instruction parallelism.
//! Their unrounded sums and corrections are merged with TwoSum before rounding.
//! TwoSum also extracts every addition residual. If g=(n+8)*u/(1-(n+8)*u), the
//! residuals have absolute mass <=g times the positive term mass. Summing
//! those residuals in working precision therefore loses <=g² times that
//! mass. 4*g² times the computed sum is an outward bound when g<1/4.
//! Rounding the corrected sum adds u times its magnitude. This tier requires
//! n<=2^24; longer inputs use the other tiers, without an API length cap.

use super::{
    CertificationTarget, Expansion, PositiveLogarithm, PositiveSum, add_up, destination_ulp, logit,
    mul_up,
};
use crate::{RealScalar, SciOxideError};

const U: f64 = 1.0 / (1_u64 << 53) as f64;
const MINIMUM: f64 = f64::from_bits(1);

#[derive(Clone, Copy)]
pub(crate) struct PositiveInputs {
    pub(crate) max: f64,
    pub(crate) active: usize,
    pub(crate) centered: bool,
}

pub(crate) fn positive_inputs<S: RealScalar>(
    values: impl Iterator<Item = S>,
) -> Result<PositiveInputs, SciOxideError> {
    let mut inputs = PositiveInputs {
        max: f64::NEG_INFINITY,
        active: 0,
        centered: true,
    };
    for x in values {
        if let super::Logit::Finite(x) = logit(x)? {
            inputs.active += 1;
            // Validation has excluded NaN, so an ordered comparison suffices;
            // IEEE max's NaN-selection machinery adds a serial dependency here.
            if x > inputs.max {
                inputs.max = x;
            }
            if inputs.centered {
                inputs.centered = x.abs() <= 1.0 / (1_u64 << 27) as f64;
            }
        }
    }
    Ok(inputs)
}

#[derive(Default, Clone, Copy)]
pub(super) struct Compensated {
    sum: f64,
    correction: f64,
}

impl Compensated {
    #[inline]
    pub(super) fn add(&mut self, term: f64) {
        let (sum, residual) = crate::arithmetic::two_sum(self.sum, term);
        self.correction += residual;
        self.sum = sum;
    }

    pub(super) fn value(self) -> f64 {
        self.sum + self.correction
    }
}

// Masking is handled by the caller; overflowing negative differences and
// shifts below -745 have exponential mass < one minimum subnormal.
#[inline]
pub(super) fn exponential(x: f64, max: f64) -> f64 {
    let shift = x - max;
    if shift < -745.0 {
        return 0.0;
    }
    let (hi, lo) = crate::arithmetic::two_sum(x, -max);
    crate::math::positive_exp(hi, lo)
}

pub(crate) struct WorkingSum {
    pub(crate) sum: f64,
    pub(crate) shifted_log: f64,
    logarithm: f64,
    bound: f64,
    shifted_bound: f64,
    relative_bound: f64,
    admitted: bool,
    cancellation: bool,
}

impl WorkingSum {
    pub(crate) fn certifies<S: RealScalar>(&self, target: CertificationTarget) -> bool {
        if !self.admitted {
            return false;
        }
        let (value, bound, ulps) = match target {
            CertificationTarget::Logarithm => {
                if self.cancellation {
                    return false;
                }
                (self.logarithm, self.bound, 14.0)
            }
            // Eight shifted-log ULP leave room for shift, subtraction and
            // destination rounding. For probabilities, numerator error <=4u,
            // denominator error <=7u and reciprocal/multiplication <=2u leave
            // three ULP of slack (subnormal rounding is bounded absolutely).
            CertificationTarget::Normalization => {
                if self.relative_bound > 7.0 * U {
                    return false;
                }
                (self.shifted_log, self.shifted_bound, 8.0)
            }
        };
        let low = RealScalar::next_after(value.abs() - bound, 0.0);
        low >= S::MIN_POSITIVE.into_f64()
            && bound.is_finite()
            && bound <= ulps * destination_ulp::<S>(low)
    }

    pub(super) fn finish(self, anchor: f64) -> PositiveSum {
        PositiveSum {
            anchor,
            shifted_log: Expansion::from(self.shifted_log),
            logarithm: PositiveLogarithm::Working(self.logarithm),
        }
    }
}

// The sink may stage probabilities in existing caller storage; it receives
// one value per input, including zero for masks. No output is published here.
pub(crate) fn working_sum<S: RealScalar>(
    mut values: impl Iterator<Item = S>,
    inputs: PositiveInputs,
    mut stage: impl FnMut(f64) -> Result<(), SciOxideError>,
) -> Result<WorkingSum, SciOxideError> {
    let mut sums = [Compensated::default(); 4];
    let mut omitted_maximum = false;
    while let Some(first) = values.next() {
        let block = [Some(first), values.next(), values.next(), values.next()]
            .map(|x| x.map(RealScalar::into_f64));
        // Compute the independent exponentials before the dependent reductions.
        // Spell out the four calls: array::map can outline its drain callback,
        // preventing the compiler from scheduling the kernels together. Padding
        // is a mask and is never passed to the staging sink.
        let terms = [
            exponential(block[0].unwrap_or(f64::NEG_INFINITY), inputs.max),
            exponential(block[1].unwrap_or(f64::NEG_INFINITY), inputs.max),
            exponential(block[2].unwrap_or(f64::NEG_INFINITY), inputs.max),
            exponential(block[3].unwrap_or(f64::NEG_INFINITY), inputs.max),
        ];
        for ((x, term), sum) in block.into_iter().zip(terms).zip(&mut sums) {
            if let Some(x) = x {
                stage(term)?;
                let omit = !omitted_maximum && x == inputs.max;
                omitted_maximum |= omit;
                sum.add(if omit { 0.0 } else { term });
            }
        }
    }
    // Preserve every component: rounding each partial sum before merging would
    // introduce first-order error outside the compensated mass bound.
    let mut sum = Compensated::default();
    for partial in sums {
        sum.add(partial.sum);
        sum.add(partial.correction);
    }
    Ok(finish(sum.value(), inputs))
}

fn finish(r: f64, inputs: PositiveInputs) -> WorkingSum {
    let n = RealScalar::next_after(inputs.active as f64, f64::INFINITY);
    // Eight merge additions also contribute residuals. Their absolute mass,
    // including the signed corrections, is covered by the conservative 4*g².
    let nu = mul_up(add_up(n, 8.0), U);
    let denominator = RealScalar::next_after(1.0 - nu, 0.0);
    let gamma = RealScalar::next_after(nu / denominator, f64::INFINITY);
    let sum_error = add_up(U, mul_up(4.0, mul_up(gamma, gamma)));
    let mass_error = add_up(
        mul_up(add_up(4.0 * U, sum_error), r),
        mul_up(n, 4.0 * MINIMUM),
    );
    let sum = 1.0 + r;
    let lower = RealScalar::next_after(sum - mass_error - U * sum, 0.0);
    let log_error = RealScalar::next_after(mass_error / lower, f64::INFINITY);
    let shifted_log = crate::math::positive_log1p(r);
    let shifted_bound = add_up(log_error, destination_ulp::<f64>(shifted_log));
    let logarithm = inputs.max + shifted_log;
    let bound = add_up(shifted_bound, destination_ulp::<f64>(logarithm));
    let relative_bound = RealScalar::next_after(
        add_up(mass_error, 0.5 * destination_ulp::<f64>(sum)) / lower,
        f64::INFINITY,
    );
    // Severe reconstruction cancellation is reserved for the existing kernels,
    // even when wider working precision could certify a binary32 destination.
    let cancellation =
        logarithm.abs() <= (1.0 + inputs.max.abs() + shifted_log) * (1.0 / (1_u64 << 20) as f64);
    WorkingSum {
        sum,
        shifted_log,
        logarithm,
        bound,
        shifted_bound,
        relative_bound,
        admitted: inputs.active <= 1 << 24 && lower > 0.0,
        cancellation,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn typical<S: RealScalar>() {
        for size in [2, 3, 8, 16, 17, 32, 64, 256, 4096] {
            let values: alloc::vec::Vec<S> = (0..size)
                .map(|i| {
                    if i % 7 == 3 {
                        S::NEG_INFINITY
                    } else {
                        S::try_from_f64_lossy((f64::from((i * 37 % 64) as u32) - 32.0) * 0.125)
                            .unwrap()
                    }
                })
                .collect();
            let inputs = positive_inputs(values.iter().copied()).unwrap();
            let working = working_sum(values.iter().copied(), inputs, |_| Ok(())).unwrap();
            assert!(
                working.certifies::<S>(CertificationTarget::Logarithm),
                "logarithm n={size}"
            );
            assert!(
                working.certifies::<S>(CertificationTarget::Normalization),
                "normalization n={size}"
            );
            let exact = super::super::exact_positive_sum(
                values.iter().copied(),
                inputs.max,
                inputs.centered,
            )
            .unwrap();
            let error = exact
                .logarithm
                .expansion()
                .sub(Expansion::from(working.logarithm))
                .absolute_upper();
            assert!(
                error <= working.bound,
                "log bound n={size}: {error:e} > {:e}",
                working.bound
            );
            let error = exact
                .shifted_log
                .sub(Expansion::from(working.shifted_log))
                .absolute_upper();
            assert!(error <= working.shifted_bound, "shifted bound n={size}");
            let actual = working.finish(inputs.max).logarithm::<S>().unwrap();
            let reference = exact.logarithm::<S>().unwrap();
            assert!(
                (actual.into_f64() - reference.into_f64()).abs()
                    <= 16.0 * destination_ulp::<S>(reference.into_f64())
            );
        }
        for size in [2, 16, 64] {
            let values = alloc::vec![S::ZERO; size];
            let inputs = positive_inputs(values.iter().copied()).unwrap();
            let working = working_sum(values.into_iter(), inputs, |_| Ok(())).unwrap();
            assert!(working.certifies::<S>(CertificationTarget::Logarithm));
            assert!(working.certifies::<S>(CertificationTarget::Normalization));
        }
    }

    #[test]
    fn typical_and_short_inputs_certify_in_both_widths() {
        typical::<f32>();
        typical::<f64>();
    }

    fn fallbacks<S: RealScalar>() {
        for values in [
            [S::try_from_f64_lossy(-core::f64::consts::LN_2).unwrap(); 2],
            [S::ZERO, S::try_from_f64_lossy(-100.0).unwrap()],
            [S::MIN_SUBNORMAL, -S::MAX],
        ] {
            let inputs = positive_inputs(values.into_iter()).unwrap();
            let working = working_sum(values.into_iter(), inputs, |_| Ok(())).unwrap();
            assert!(!working.certifies::<S>(CertificationTarget::Logarithm));
            let actual =
                super::super::positive_sum(values.into_iter(), CertificationTarget::Logarithm)
                    .unwrap();
            let exact =
                super::super::exact_positive_sum(values.into_iter(), inputs.max, inputs.centered)
                    .unwrap();
            let reference = exact.logarithm::<S>().unwrap();
            let actual = actual.logarithm::<S>().unwrap();
            assert!(
                (actual.into_f64() - reference.into_f64()).abs()
                    <= 16.0 * destination_ulp::<S>(reference.into_f64())
            );
        }
        let inputs = PositiveInputs {
            max: 0.0,
            active: 1 << 25,
            centered: true,
        };
        let working = finish((inputs.active - 1) as f64, inputs);
        assert!(!working.certifies::<S>(CertificationTarget::Logarithm));
        assert!(!working.certifies::<S>(CertificationTarget::Normalization));
        let boundary = PositiveInputs {
            active: 1 << 24,
            ..inputs
        };
        assert!(
            finish((boundary.active - 1) as f64, boundary)
                .certifies::<S>(CertificationTarget::Normalization)
        );
        // Normalization is independent of full-logarithm cancellation.
        let values = [S::try_from_f64_lossy(-core::f64::consts::LN_2).unwrap(); 2];
        let inputs = positive_inputs(values.into_iter()).unwrap();
        assert!(
            working_sum(values.into_iter(), inputs, |_| Ok(()))
                .unwrap()
                .certifies::<S>(CertificationTarget::Normalization)
        );
    }

    #[test]
    fn ill_conditioned_subnormal_and_huge_counts_use_existing_tiers() {
        fallbacks::<f32>();
        fallbacks::<f64>();
    }

    fn certificate_boundary<S: RealScalar>() {
        let ulp = destination_ulp::<S>(1.0);
        let mut candidate = finish(
            1.0,
            PositiveInputs {
                max: 2.0,
                active: 2,
                centered: false,
            },
        );
        candidate.logarithm = 1.0 + 32.0 * ulp;
        candidate.bound = 14.0 * ulp;
        assert!(candidate.certifies::<S>(CertificationTarget::Logarithm));
        candidate.bound = candidate.bound.next_after(f64::INFINITY);
        assert!(!candidate.certifies::<S>(CertificationTarget::Logarithm));
        candidate.logarithm = 1.0;
        candidate.bound = 8.0 * ulp;
        // The lower endpoint is below the binade, whose grid is half as wide.
        assert!(!candidate.certifies::<S>(CertificationTarget::Logarithm));
        candidate.shifted_log = 1.0 + 32.0 * ulp;
        candidate.shifted_bound = 8.0 * ulp;
        candidate.relative_bound = 7.0 * U;
        assert!(candidate.certifies::<S>(CertificationTarget::Normalization));
        candidate.relative_bound = candidate.relative_bound.next_after(f64::INFINITY);
        assert!(!candidate.certifies::<S>(CertificationTarget::Normalization));
    }

    #[test]
    fn certificates_use_outward_bounds_and_the_lower_destination_grid() {
        certificate_boundary::<f32>();
        certificate_boundary::<f64>();
    }

    fn fixture_admission<S: RealScalar>() {
        let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
        let mut counts = [0; 2];
        for row in include_str!("../../tests/data/normalization_fast.txt").lines() {
            let fields: alloc::vec::Vec<_> = row.split_whitespace().collect();
            if fields[0] != width || !fields[1].starts_with("certificate-") {
                continue;
            }
            let mut values = alloc::vec::Vec::new();
            for group in fields[3..].chunks_exact(4) {
                let bits = u64::from_str_radix(group[1], 16).unwrap();
                let value = if width == "32" {
                    f64::from(f32::from_bits(bits as u32))
                } else {
                    f64::from_bits(bits)
                };
                let value = if value == f64::NEG_INFINITY {
                    S::NEG_INFINITY
                } else {
                    S::try_from_f64_lossy(value).unwrap()
                };
                values.extend(core::iter::repeat_n(
                    value,
                    group[0].parse::<usize>().unwrap(),
                ));
            }
            let inputs = positive_inputs(values.iter().copied()).unwrap();
            let working = working_sum(values.into_iter(), inputs, |_| Ok(())).unwrap();
            let admitted = fields[1] == "certificate-admit";
            assert_eq!(
                working.certifies::<S>(CertificationTarget::Normalization),
                admitted,
                "{row}: shifted log={:e}, bound={:e}, relative={:e}",
                working.shifted_log,
                working.shifted_bound,
                working.relative_bound
            );
            counts[usize::from(admitted)] += 1;
        }
        assert_eq!(counts, [3, 3]);
    }

    #[test]
    fn independent_fixtures_straddle_normalization_admission_in_both_widths() {
        fixture_admission::<f32>();
        fixture_admission::<f64>();
    }

    #[test]
    fn shift_exponentials_cover_tables_reduction_boundaries_and_tails() {
        // All table entries, together with Cody-Waite boundaries spanning
        // the entire normal scaling range. Adjacent values can change k.
        for k in (-258_048..=0).step_by(1024).chain(-256..=0) {
            let boundary = (f64::from(k) + 0.5) * (core::f64::consts::LN_2 / 256.0);
            for x in [
                boundary.next_after(f64::NEG_INFINITY),
                boundary,
                boundary.next_after(0.0),
            ] {
                check_exponential(x.min(0.0), 0.0);
                check_exponential(x.min(0.0) + 50.0, 50.0);
            }
        }
        for x in [-700.0, -708.0, -740.0, -745.0, -746.0] {
            for x in [x.next_after(f64::NEG_INFINITY), x, x.next_after(0.0)] {
                check_exponential(x, 0.0);
            }
        }
        assert_eq!(exponential(-f64::MAX, f64::MAX), 0.0);
        assert_eq!(exponential(f64::MAX, f64::MAX), 1.0);
    }

    fn check_exponential(x: f64, max: f64) {
        let actual = exponential(x, max);
        let expected = Expansion::from(x).sub(Expansion::from(max)).exp();
        let bound = add_up(mul_up(4.0 * U, actual), 4.0 * MINIMUM);
        assert!(
            expected.sub(Expansion::from(actual)).absolute_upper() <= bound,
            "x={x:e}, max={max:e}"
        );
    }
}
