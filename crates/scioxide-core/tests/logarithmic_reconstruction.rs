//! Independent logarithmic reconstruction and consumer regressions.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::*;
use std::collections::BTreeMap;

fn decode<S: RealScalar>(bits: &str) -> S {
    let bits = u64::from_str_radix(bits, 16).unwrap();
    let value = if S::MANTISSA_DIGITS == 24 {
        f32::from_bits(bits as u32) as f64
    } else {
        f64::from_bits(bits)
    };
    if value == f64::NEG_INFINITY {
        S::NEG_INFINITY
    } else {
        S::try_from_f64_lossy(value).unwrap()
    }
}

fn fixtures<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let mut maxima = BTreeMap::<&str, (f64, String, usize)>::new();
    let mut strata = BTreeMap::<&str, usize>::new();
    let mut failures = 0;
    for line in include_str!("data/logarithmic.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        *strata.entry(fields[2]).or_default() += 1;
        let expected = (fields[3] != "zero").then(|| decode::<S>(fields[3]));
        let phase = Complex::new(decode::<S>(fields[4]), decode::<S>(fields[5]));
        let condition = f64::from_bits(u64::from_str_radix(fields[6], 16).unwrap());
        let groups = fields[7].parse::<usize>().unwrap();
        assert_eq!(fields.len(), 8 + 5 * groups);
        let mut values = Vec::new();
        let mut weights = Vec::new();
        for group in fields[8..].chunks_exact(5) {
            let count = group[0].parse::<usize>().unwrap();
            values.extend(core::iter::repeat_n(
                Complex::new(decode::<S>(group[1]), decode::<S>(group[2])),
                count,
            ));
            weights.extend(core::iter::repeat_n(
                Complex::new(decode::<S>(group[3]), decode::<S>(group[4])),
                count,
            ));
        }
        let real: Vec<_> = values.iter().map(|v| v.re).collect();
        let real_weights: Vec<_> = weights.iter().map(|v| v.re).collect();
        let shape = [values.len()];
        let layout = StridedLayout::new(&shape, &[1]).unwrap();
        let strict = fields[1] == "positive"
            || fields[2] == "common-phase"
            || fields[2] == "signed-unit-sum"
            || (fields[1] == "weighted"
                && weights.iter().all(|w| w.im == S::ZERO && w.re >= S::ZERO));
        let mut check = |name, log: Option<S>, actual_phase: Complex<S>| {
            let entry = maxima.entry(name).or_insert((0.0, String::new(), 0));
            entry.2 += 1;
            let valid = match (expected, log) {
                (None, None) => actual_phase == Complex::new(S::ZERO, S::ZERO),
                (Some(e), Some(a)) => {
                    let magnitude = e.abs();
                    let next = magnitude.next_after(S::INFINITY);
                    let ulp = if next.is_finite() {
                        next - magnitude
                    } else {
                        magnitude
                            - magnitude.next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)
                    }
                    .into_f64();
                    let error = (a.into_f64() - e.into_f64()).abs();
                    let ulps = error / ulp;
                    if ulps > entry.0 || entry.1.is_empty() {
                        entry.0 = ulps;
                        entry.1 = line.to_owned();
                    }
                    let bound = if strict {
                        16.0 * ulp
                    } else {
                        64.0 * S::EPSILON.into_f64() * condition * (1.0 + e.into_f64().abs())
                            + 8.0 * S::MIN_SUBNORMAL.into_f64()
                    };
                    let phase_error = (actual_phase.re.into_f64() - phase.re.into_f64())
                        .abs()
                        .max((actual_phase.im.into_f64() - phase.im.into_f64()).abs());
                    let zero_sign =
                        e != S::ZERO || a.into_f64().to_bits() == e.into_f64().to_bits();
                    error <= bound
                        && phase_error <= 64.0 * S::EPSILON.into_f64() * condition
                        && zero_sign
                        && (e == S::ZERO || a != S::ZERO || condition > 16.0)
                }
                _ => false,
            };
            if !valid {
                failures += 1;
                if failures <= 20 {
                    eprintln!("{name} {line}: log={log:?}, phase={actual_phase:?}");
                }
            }
        };
        let one = Complex::new(S::ONE, S::ZERO);
        let check_angle = |angle: S| {
            let reference = phase.im.atan2(phase.re);
            assert!(
                (angle.into_f64() - reference.into_f64()).abs()
                    <= 64.0 * S::EPSILON.into_f64() * condition,
                "principal phase {line}: {angle:?} != {reference:?}"
            );
        };
        if fields[1] == "positive" {
            let v = logsumexp_value(&real).unwrap();
            check(
                "positive-value",
                v.log_magnitude(),
                Complex::new(v.sign(), S::ZERO),
            );
            if values.iter().all(|v| v.re == S::NEG_INFINITY) {
                assert_eq!(logsumexp(&real).unwrap(), S::NEG_INFINITY);
            } else {
                check("positive", Some(logsumexp(&real).unwrap()), one);
                if values.len() == 2 {
                    check("logaddexp", Some(logaddexp(real[0], real[1]).unwrap()), one);
                }
                let result_layout = StridedLayout::new(&[], &[]).unwrap();
                let mut out = [S::ZERO];
                let mut scratch = [S::ZERO];
                scioxide_core::logsumexp_axis_into(
                    NdView::new(&real, layout).unwrap(),
                    0,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                check("positive-axis", Some(out[0]), one);
            }
        }
        if fields[1] != "complex" {
            if weights.iter().all(|w| w.im == S::ZERO) {
                let v = weighted_logsumexp(&real, &real_weights).unwrap();
                check(
                    "weighted-real",
                    v.log_magnitude(),
                    Complex::new(v.sign(), S::ZERO),
                );
                let result_layout = StridedLayout::new(&[], &[]).unwrap();
                let mut out = [LogValue::<S, S>::ZERO];
                let mut scratch = out;
                scioxide_core::weighted_logsumexp_axis_into(
                    NdView::new(&real, layout).unwrap(),
                    NdView::new(&real_weights, layout).unwrap(),
                    0,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                check(
                    "weighted-real-axis",
                    out[0].log_magnitude(),
                    Complex::new(out[0].sign(), S::ZERO),
                );
            }
            let v = weighted_logsumexp(&real, &weights).unwrap();
            check(
                "weighted-complex",
                v.log_magnitude(),
                v.scaled_phase().round().unwrap(),
            );
            let result_layout = StridedLayout::new(&[], &[]).unwrap();
            let mut out = [LogValue::ZERO];
            let mut scratch = out;
            scioxide_core::weighted_logsumexp_axis_into(
                NdView::new(&real, layout).unwrap(),
                NdView::new(&weights, layout).unwrap(),
                0,
                scioxide_core::ExecutionPolicy::Sequential,
                NdViewMut::new(&mut out, result_layout).unwrap(),
                &mut scratch,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            check(
                "weighted-axis",
                out[0].log_magnitude(),
                out[0].scaled_phase().round().unwrap(),
            );
        }
        if fields[1] != "weighted" {
            let v = logsumexp_value(&values).unwrap();
            check(
                "complex-value",
                v.log_magnitude(),
                v.scaled_phase().round().unwrap(),
            );
            if expected.is_none() {
                if values.iter().all(|v| v.re == S::NEG_INFINITY) {
                    assert_eq!(
                        logsumexp(&values).unwrap(),
                        Complex::new(S::NEG_INFINITY, S::ZERO)
                    );
                } else {
                    assert!(logsumexp(&values).is_err());
                }
            } else {
                let z = logsumexp(&values).unwrap();
                check_angle(z.im);
                check("complex", Some(z.re), v.scaled_phase().round().unwrap());
                if values.len() == 2 {
                    let z = logaddexp(values[0], values[1]).unwrap();
                    check_angle(z.im);
                    check(
                        "complex-logaddexp",
                        Some(z.re),
                        v.scaled_phase().round().unwrap(),
                    );
                }
                let result_layout = StridedLayout::new(&[], &[]).unwrap();
                let mut out = [Complex::new(S::ZERO, S::ZERO)];
                let mut scratch = out;
                scioxide_core::logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    0,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                check_angle(out[0].im);
                check(
                    "complex-axis",
                    Some(out[0].re),
                    v.scaled_phase().round().unwrap(),
                );
            }
        }
    }
    assert_eq!(
        strata["all-exponent-bins"],
        if width == "32" { 277 } else { 2098 }
    );
    assert_eq!(strata["ln2-neighborhood"], 81);
    assert_eq!(strata["reduction-size"], 39);
    for name in [
        "unit-sum-surface",
        "unit-sum-dense",
        "range-boundary",
        "equal-range-boundary",
        "dominant-tiny",
        "singleton",
        "weighted-unit-sum",
        "positive-weight",
        "signed-cancellation",
        "weighted-window",
        "window-boundary",
        "common-phase",
        "phase-cancellation",
        "unequal-positive-weight",
        "conditioned-small-log",
        "translated-window",
        "ordinary-overflow",
        "symmetric-unit-sum",
        "signed-unit-sum",
    ] {
        assert!(strata[name] > 0, "missing {name}");
    }
    for (name, (error, bits, count)) in maxima {
        println!("logarithmic/{width}/{name}: count={count} max_ulp={error} worst={bits}");
    }
    assert_eq!(failures, 0, "{width}-bit logarithmic oracle failures");
}

#[test]
fn independent_f32_reconstruction() {
    fixtures::<f32>();
}

#[test]
fn independent_f64_reconstruction() {
    fixtures::<f64>();
}

#[test]
fn masked_and_weighted_reference_campaign_is_complete() {
    let mut inventory = BTreeMap::<(&str, &str), (usize, usize)>::new();
    for line in include_str!("data/logarithmic.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let count = fields[8..]
            .chunks_exact(5)
            .map(|group| group[0].parse::<usize>().unwrap())
            .sum::<usize>();
        let entry = inventory.entry((fields[0], fields[2])).or_default();
        entry.0 += 1;
        entry.1 = entry.1.max(count);
    }
    for width in ["32", "64"] {
        assert_eq!(inventory[&(width, "weighted-size")], (20, 100000));
        assert_eq!(inventory[&(width, "weighted-tier")].0, 22);
        assert_eq!(inventory[&(width, "masked")].0, 12);
    }
}

#[test]
fn rounded_negative_ln_two_keeps_its_residual() {
    let x = -core::f64::consts::LN_2;
    // MPFR 4.2.2 at 2304 and 2560 bits, with exact binary64 input.
    let expected = 2.319_046_813_846_299_6e-17_f64;
    let z = Complex::new(x, 0.0);
    for actual in [
        logaddexp(x, x).unwrap(),
        logsumexp(&[x, x]).unwrap(),
        logsumexp_value(&[x, x]).unwrap().log_magnitude().unwrap(),
        weighted_logsumexp(&[x, x], &[1.0, 1.0])
            .unwrap()
            .log_magnitude()
            .unwrap(),
        weighted_logsumexp(&[x, x], &[Complex::new(1.0, 0.0); 2])
            .unwrap()
            .log_magnitude()
            .unwrap(),
        logsumexp_value(&[z, z]).unwrap().log_magnitude().unwrap(),
        logsumexp(&[z, z]).unwrap().re,
        logaddexp(z, z).unwrap().re,
    ] {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{actual:e}");
    }
}

#[test]
fn dominant_term_keeps_subnormal_logarithms() {
    for x in [
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ] {
        assert_eq!(logsumexp(&[x]).unwrap().to_bits(), x.to_bits());
        assert_eq!(logaddexp(x, -1200.0).unwrap().to_bits(), x.to_bits());
        assert_eq!(
            weighted_logsumexp::<f64>(&[x], &[1.0])
                .unwrap()
                .log_magnitude()
                .unwrap()
                .to_bits(),
            x.to_bits()
        );
    }
    for x in [-740.0_f64, -745.0] {
        let expected = RealScalar::exp(x);
        assert_eq!(logaddexp(0.0, x).unwrap().to_bits(), expected.to_bits());
        assert_eq!(
            weighted_logsumexp::<f64>(&[0.0, x], &[1.0, 1.0])
                .unwrap()
                .log_magnitude()
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn singleton_normalization_keeps_exact_identities() {
    for x in [0.0_f64, -0.0, 1e-12, -1e-12, f64::from_bits(1), f64::MAX] {
        let mut out = [77.0];
        let mut scratch = [0.0];
        softmax_into(&[x], &mut out, &mut scratch).unwrap();
        assert_eq!(out[0].to_bits(), 1.0_f64.to_bits());
        log_softmax_into(&[x], &mut out, &mut scratch).unwrap();
        assert_eq!(out[0].to_bits(), 0.0_f64.to_bits(), "x={x:e}");
    }
}

fn resources<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let x = -S::LN_2;
    let shape = [2, 2];
    let layout = StridedLayout::new(&shape, &[2, 1]).unwrap();
    let result_layout = StridedLayout::new(&[2], &[1]).unwrap();
    let sentinel = [S::NEG_ZERO, S::MAX];
    let zero = LogValue::<S, S>::ZERO;
    let complex_sentinel = [Complex::new(S::NEG_ZERO, S::MAX); 2];
    {
        let values = [S::try_from_f64_lossy(5.0).unwrap(); 4];
        let weights = [S::ONE; 4];
        let complex_weights = weights.map(|re| Complex::new(re, S::HALF));
        let complex_values = [
            Complex::new(values[0], S::HALF),
            Complex::new(values[1], S::ONE),
            Complex::new(values[2], S::HALF),
            Complex::new(values[3], S::ONE),
        ];
        let _ = weighted_logsumexp(&values, &weights).unwrap();
        let _ = weighted_logsumexp(&values, &complex_weights).unwrap();
        let _ = logsumexp_value(&complex_values).unwrap();
        logsumexp(&complex_values).unwrap();
        let mut output = complex_sentinel;
        let mut scratch = output;
        logsumexp_axis_into(
            NdView::new(&complex_values, layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut output, result_layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let mut output = [LogValue::<Complex<S>, S>::ZERO; 2];
        let mut scratch = output;
        weighted_logsumexp_axis_into(
            NdView::new(&values, layout).unwrap(),
            NdView::new(&complex_weights, layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut output, result_layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        logaddexp(x, x).unwrap();
        let _ = logsumexp_value(&[x, x]).unwrap();
        let _ = weighted_logsumexp(&[x, x], &[S::ONE, S::ONE]).unwrap();
        logsumexp(&[Complex::new(x, S::ZERO); 2]).unwrap();
        let _ = weighted_logsumexp(&[x, x], &[Complex::new(S::ONE, S::ZERO); 2]).unwrap();
        for bad in [S::NAN, S::INFINITY] {
            assert_eq!(
                logaddexp(x, bad).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
            assert!(weighted_logsumexp(&[bad], &[S::ZERO]).is_err());
            assert!(weighted_logsumexp(&[bad], &[Complex::new(S::ZERO, S::ZERO)]).is_err());
            assert!(weighted_logsumexp(&[x], &[bad]).is_err());
            assert!(logsumexp_value(&[Complex::new(x, bad)]).is_err());
            let values = [x, x, x, bad];
            let mut out = sentinel;
            let mut scratch = [S::ZERO; 2];
            assert!(
                scioxide_core::logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    1,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
            );
            assert_eq!(
                out.map(|v| v.into_f64().to_bits()),
                sentinel.map(|v| v.into_f64().to_bits())
            );
            let weights = [S::ONE, S::ONE, S::ZERO, S::ZERO];
            let mut out = [zero; 2];
            let mut scratch = out;
            assert!(
                scioxide_core::weighted_logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    NdView::new(&weights, layout).unwrap(),
                    1,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
            );
            assert!(out.iter().all(|v| v.log_magnitude().is_none()));
            let values = values.map(|re| Complex::new(re, S::ZERO));
            let mut out = complex_sentinel;
            let mut scratch = out;
            assert!(
                scioxide_core::logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    1,
                    scioxide_core::ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut out, result_layout).unwrap(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
            );
            assert_eq!(
                out.map(|v| (v.re.into_f64().to_bits(), v.im.into_f64().to_bits())),
                complex_sentinel.map(|v| (v.re.into_f64().to_bits(), v.im.into_f64().to_bits()))
            );
        }
        let values = [x; 4];
        let mut out = sentinel;
        let mut scratch = [S::ZERO; 2];
        assert!(
            scioxide_core::logsumexp_axis_into(
                NdView::new(&values, layout).unwrap(),
                1,
                scioxide_core::ExecutionPolicy::Sequential,
                NdViewMut::new(&mut out, result_layout).unwrap(),
                &mut scratch[..1],
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(
            out.map(|v| v.into_f64().to_bits()),
            sentinel.map(|v| v.into_f64().to_bits())
        );
        scioxide_core::logsumexp_axis_into(
            NdView::new(&values, layout).unwrap(),
            1,
            scioxide_core::ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, result_layout).unwrap(),
            &mut scratch,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let weights = [S::ONE; 4];
        let mut out = [zero; 2];
        let mut scratch = out;
        scioxide_core::weighted_logsumexp_axis_into(
            NdView::new(&values, layout).unwrap(),
            NdView::new(&weights, layout).unwrap(),
            1,
            scioxide_core::ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, result_layout).unwrap(),
            &mut scratch,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let values = values.map(|re| Complex::new(re, S::ZERO));
        let mut out = complex_sentinel;
        let mut scratch = out;
        scioxide_core::logsumexp_axis_into(
            NdView::new(&values, layout).unwrap(),
            1,
            scioxide_core::ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, result_layout).unwrap(),
            &mut scratch,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(logsumexp::<S>(&[]).unwrap(), S::NEG_INFINITY);
        assert!(logsumexp_value::<S>(&[]).unwrap().log_magnitude().is_none());
        assert!(weighted_logsumexp::<S>(&[x], &[]).is_err());
        assert!(
            weighted_logsumexp(&[x, x], &[S::ONE, -S::ONE])
                .unwrap()
                .log_magnitude()
                .is_none()
        );
    };
}

#[test]
fn allocation_and_failure_publication_contracts() {
    resources::<f32>();
    resources::<f64>();
}

fn axis_context(context: ExecutionContext<'_>, policy: ExecutionPolicy) {
    let x = -core::f64::consts::LN_2;
    let values = [x, x, 0.0, 0.0, 1.0, 1.0];
    let expected = [
        2.319_046_813_846_299_6e-17,
        core::f64::consts::LN_2,
        1.0 + core::f64::consts::LN_2,
    ];
    let layout = StridedLayout::new(&[2, 3], &[1, 2]).unwrap();
    let weights_layout = StridedLayout::new_broadcast(&[2, 3], &[1, 0]).unwrap();
    let output_layout = StridedLayout::new(&[3], &[2]).unwrap();
    let mut out = [-77.0; 5];
    let mut scratch = [0.0; 3];
    scioxide_core::logsumexp_axis_into(
        NdView::new(&values, layout).unwrap(),
        0,
        policy,
        NdViewMut::new(&mut out, output_layout).unwrap(),
        &mut scratch,
        context,
    )
    .unwrap();
    assert_eq!(out, [expected[0], -77.0, expected[1], -77.0, expected[2]]);
    let zero = LogValue::<f64, f64>::ZERO;
    let mut out = [zero; 5];
    let mut scratch = [zero; 3];
    scioxide_core::weighted_logsumexp_axis_into(
        NdView::new(&values, layout).unwrap(),
        NdView::new(&[1.0, 1.0], weights_layout).unwrap(),
        0,
        policy,
        NdViewMut::new(&mut out, output_layout).unwrap(),
        &mut scratch,
        context,
    )
    .unwrap();
    assert_eq!(
        out.map(|v| v.log_magnitude()),
        [
            Some(expected[0]),
            None,
            Some(expected[1]),
            None,
            Some(expected[2])
        ]
    );
    let values = values.map(|re| Complex::new(re, 0.25));
    let mut out = [Complex::new(-77.0, -77.0); 5];
    let mut scratch = [Complex::new(0.0, 0.0); 3];
    scioxide_core::logsumexp_axis_into(
        NdView::new(&values, layout).unwrap(),
        0,
        policy,
        NdViewMut::new(&mut out, output_layout).unwrap(),
        &mut scratch,
        context,
    )
    .unwrap();
    assert_eq!(
        out.map(|v| v.re),
        [expected[0], -77.0, expected[1], -77.0, expected[2]]
    );
}

#[test]
fn strided_contexts_and_broadcast_weights() {
    axis_context(ExecutionContext::sequential(), ExecutionPolicy::Sequential);
    #[cfg(feature = "std")]
    {
        let workers = core::num::NonZeroUsize::new(2).unwrap();
        let pool = ThreadExecutor::new(workers);
        axis_context(
            ExecutionContext::new(&pool),
            ExecutionPolicy::Parallel(workers),
        );
    }
}

enum Fault {
    Omit,
    Replay,
}

impl Executor for Fault {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Omit => Ok(()),
            Self::Replay => {
                SequentialExecutor.execute(context, plan, operation)?;
                SequentialExecutor.execute(context, plan, operation)
            }
        }
    }
}

fn failed_context(context: ExecutionContext<'_>) {
    let input_layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let output_layout = StridedLayout::new(&[2], &[1]).unwrap();
    let values = [-core::f64::consts::LN_2; 4];
    let mut out = [77.0; 2];
    let mut scratch = [0.0; 2];
    assert!(
        logsumexp_axis_into(
            NdView::new(&values, input_layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, output_layout).unwrap(),
            &mut scratch,
            context
        )
        .is_err()
    );
    assert_eq!(out, [77.0; 2]);
    for logarithmic in [false, true] {
        let mut out = [77.0; 4];
        let mut scratch = [0.0; 4];
        let input = NdView::new(&values, input_layout).unwrap();
        let result = NdViewMut::new(&mut out, input_layout).unwrap();
        let policy = ExecutionPolicy::Sequential;
        let failure = if logarithmic {
            log_softmax_axis_into(input, 1, policy, result, &mut scratch, context)
        } else {
            softmax_axis_into(input, 1, policy, result, &mut scratch, context)
        };
        assert!(failure.is_err());
        assert_eq!(out, [77.0; 4]);
    }
    let weights = [1.0; 4];
    let zero = LogValue::<f64, f64>::ZERO;
    let mut out = [zero; 2];
    let mut scratch = out;
    assert!(
        weighted_logsumexp_axis_into(
            NdView::new(&values, input_layout).unwrap(),
            NdView::new(&weights, input_layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, output_layout).unwrap(),
            &mut scratch,
            context
        )
        .is_err()
    );
    assert!(out.iter().all(|v| v.log_magnitude().is_none()));
    let values = values.map(|re| Complex::new(re, 0.25));
    let old = [Complex::new(77.0, 77.0); 2];
    let mut out = old;
    let mut scratch = old;
    assert!(
        logsumexp_axis_into(
            NdView::new(&values, input_layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out, output_layout).unwrap(),
            &mut scratch,
            context
        )
        .is_err()
    );
    assert_eq!(out, old);
}

#[test]
fn cancellation_omission_and_replay_do_not_publish() {
    let token = CancellationToken::new();
    token.cancel();
    failed_context(ExecutionContext::sequential().with_cancellation(&token));
    for fault in [Fault::Omit, Fault::Replay] {
        failed_context(ExecutionContext::new(&fault));
        let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
        for logarithmic in [false, true] {
            let input = NdView::new(&[0.0_f64; 4], layout).unwrap();
            let context = ExecutionContext::new(&fault);
            let mut output = [77.0; 4];
            let mut scratch = [0.0; 4];
            let target = NdViewMut::new(&mut output, layout).unwrap();
            let policy = ExecutionPolicy::Sequential;
            let result = if logarithmic {
                log_softmax_axis_into(input, 1, policy, target, &mut scratch, context)
            } else {
                softmax_axis_into(input, 1, policy, target, &mut scratch, context)
            };
            assert_eq!(output, [77.0; 4]);
            assert_eq!(
                result.unwrap_err().kind(),
                SciOxideErrorKind::ExecutionFailure
            );
        }
    }
}

#[test]
fn empty_lanes_and_empty_batches_are_distinct() {
    let lanes = StridedLayout::new(&[3, 0], &[0, 1]).unwrap();
    let output = StridedLayout::new(&[3], &[1]).unwrap();
    let mut out = [77.0; 3];
    let mut scratch = [0.0; 3];
    scioxide_core::logsumexp_axis_into(
        NdView::<f64>::new(&[], lanes).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut out, output).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out, [f64::NEG_INFINITY; 3]);
    let zero = LogValue::<f64, f64>::ZERO;
    let mut out = [LogValue::from_value(1.0).unwrap(); 3];
    let mut scratch = [zero; 3];
    scioxide_core::weighted_logsumexp_axis_into(
        NdView::<f64>::new(&[], lanes).unwrap(),
        NdView::<f64>::new(&[], lanes).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut out, output).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(out.iter().all(|v| v.log_magnitude().is_none()));
    let batches = StridedLayout::new(&[0, 2], &[2, 1]).unwrap();
    let output = StridedLayout::new(&[0], &[1]).unwrap();
    scioxide_core::logsumexp_axis_into(
        NdView::<f64>::new(&[], batches).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut [], output).unwrap(),
        &mut [],
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    scioxide_core::logsumexp_axis_into(
        NdView::<Complex64>::new(&[], batches).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut [], output).unwrap(),
        &mut [],
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
}
