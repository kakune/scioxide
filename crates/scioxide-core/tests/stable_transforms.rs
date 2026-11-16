#![allow(clippy::panic)]
use scioxide_core::*;

#[test]
fn real_limits_and_domains() {
    assert_eq!(exprel(0.0_f64).unwrap(), 1.0);
    assert_eq!(sinc(0.0_f64).unwrap(), 1.0);
    assert_eq!(sincpi(0.0_f32).unwrap(), 1.0);
    assert_eq!(xlogy(0.0_f64, -1.0).unwrap(), 0.0);
    assert_eq!(xlog1py(0.0_f32, -2.0).unwrap(), 0.0);
    assert!(logit(0.0_f64).is_err());
    assert!(logit(1.0_f32).is_err());
    assert!(xlogy(1.0_f64, 0.0).is_err());
    assert!(powm1(-2.0_f64, 0.5).is_err());
    assert!(sigmoid(f64::NAN).is_err());
    assert!(sin_pi(f32::INFINITY).is_err());
}

#[test]
fn stable_tails_and_corrections() {
    assert_eq!(softplus(1000.0_f64).unwrap(), 1000.0);
    assert_eq!(log_sigmoid(-1000.0_f64).unwrap(), -1000.0);
    assert!(sigmoid(-740.0_f64).unwrap() > 0.0);
    assert!(exprel(714.0_f64).unwrap().is_finite());
    assert!(cosm1(1e-12_f64).unwrap() < 0.0);
    assert!(powm1(1.0 + f64::EPSILON, 0.5).unwrap() > 0.0);
    assert_eq!(logaddexp(f64::MAX, -f64::MAX).unwrap(), f64::MAX);
    assert_eq!(powm1(-2.0_f64, 3.0).unwrap(), -9.0);
}

#[test]
fn exact_phases_both_widths() {
    for x in [-1e30, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1e30] {
        assert_eq!(sin_pi(x).unwrap(), 0.0);
        assert_eq!(sin_pi(x as f32).unwrap(), 0.0);
        if x != 0.0 {
            assert_eq!(sincpi(x).unwrap(), 0.0);
        }
    }
    for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
        assert_eq!(cos_pi(x).unwrap(), 0.0);
        assert_eq!(cos_pi(x as f32).unwrap(), 0.0);
        assert!(tan_pi(x).is_err());
    }
    assert_eq!(sin_degrees(90.0_f64).unwrap(), 1.0);
    assert_eq!(cos_degrees(180.0_f32).unwrap(), -1.0);
    assert_eq!(sin_pi(-0.0_f64).unwrap().to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn logarithmic_zero_and_cancellation() {
    assert_eq!(logsumexp_value::<f64>(&[]).unwrap().log_magnitude(), None);
    let z = weighted_logsumexp::<f64>(&[10000.0, 10000.0, -10000.0], &[1.0, -1.0, 2.0]).unwrap();
    assert_eq!(z.scaled_phase().try_value().unwrap(), 1.0);
    assert!((z.log_magnitude().unwrap() - (-10000.0 + 2.0_f64.ln())).abs() < 2e-12);
    let z = weighted_logsumexp::<f64>(&[0.0, -1e-20], &[1.0, -1.0]).unwrap();
    assert!((z.log_magnitude().unwrap() - 1e-20_f64.ln()).abs() < 1e-14);
    let z = weighted_logsumexp::<f32>(&[0.0_f32, 0.0], &[1.0, -1.0]).unwrap();
    assert_eq!(z.log_magnitude(), None);
    let z = weighted_logsumexp(
        &[1000.0, 1000.0],
        &[Complex::new(1.0, 2.0), Complex::new(-1.0, -1.0)],
    )
    .unwrap();
    assert_eq!(
        z.scaled_phase().try_value().unwrap(),
        Complex::new(0.0, 1.0)
    );
    assert_eq!(z.log_magnitude(), Some(1000.0));
}

#[test]
fn normalization_and_transaction() {
    let x = [1000.0, 1001.0, 999.0];
    let mut a = [0.0; 3];
    let mut b = [0.0; 3];
    let mut scratch = [0.0; 3];
    softmax_into(&x, &mut a, &mut scratch).unwrap();
    softmax_into(&[0.0, 1.0, -1.0], &mut b, &mut scratch).unwrap();
    assert_eq!(a, b);
    assert!((a.iter().sum::<f64>() - 1.0).abs() < 3e-16);
    log_softmax_into(&x, &mut b, &mut scratch).unwrap();
    for (p, l) in a.iter().zip(b) {
        assert!((p.ln() - l).abs() < 5e-16);
    }
    let old = a;
    assert!(softmax_into(&[0.0, f64::NAN, 1.0], &mut a, &mut scratch).is_err());
    assert_eq!(a, old);
    assert!(softmax_into(&x, &mut a, &mut scratch[..2]).is_err());
    assert_eq!(a, old);
}

#[test]
fn axis_layouts_and_empty_batches() {
    let shape = [2, 3];
    let layout = StridedLayout::new(&shape, &[3, 1]).unwrap();
    let x = [1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0];
    let mut result = [0.0; 6];
    let mut scratch = [0.0; 6];
    scioxide_core::softmax_axis_into(
        NdView::new(&x, layout).unwrap(),
        0,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut result, layout).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(result, [0.5; 6]);
    let transposed = StridedLayout::new(&shape, &[1, 2]).unwrap();
    scioxide_core::transform_into(
        NdView::new(&x, layout).unwrap(),
        sigmoid,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut result, transposed).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(result[2], sigmoid(2.0).unwrap());
    let layout = StridedLayout::new(&[0, 3], &[3, 1]).unwrap();
    scioxide_core::softmax_axis_into(
        NdView::new(&[], layout).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut [], layout).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn independent_mpfr_fixtures() {
    fn check<S: RealScalar>(op: &str, x: S, y: S, expected: S) {
        let actual = match op {
            "softplus" => softplus(x),
            "sigmoid" => sigmoid(x),
            "log_sigmoid" => log_sigmoid(x),
            "logit" => logit(x),
            "exprel" => exprel(x),
            "cosm1" => cosm1(x),
            "sinc" => sinc(x),
            "sincpi" => sincpi(x),
            "sin_pi" => sin_pi(x),
            "cos_pi" => cos_pi(x),
            "tan_pi" => tan_pi(x),
            "sin_degrees" => sin_degrees(x),
            "cos_degrees" => cos_degrees(x),
            "tan_degrees" => tan_degrees(x),
            "logaddexp" => logaddexp(x, y),
            "xlogy" => xlogy(x, y),
            "xlog1py" => xlog1py(x, y),
            "powm1" => powm1(x, y),
            _ => panic!("unknown fixture operation {op}"),
        }
        .unwrap_or_else(|e| panic!("{op}({x:?},{y:?}): {e}; expected {expected:?}"));
        let unit = if expected == S::ZERO {
            S::ZERO.next_after(S::ONE)
        } else {
            {
                let next = expected.abs().next_after(S::INFINITY);
                if next.is_finite() {
                    next - expected.abs()
                } else {
                    expected.abs()
                        - expected
                            .abs()
                            .next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)
                }
            }
        };
        let error = (actual.into_f64() - expected.into_f64()).abs();
        assert!(
            error <= 16.0 * unit.into_f64(),
            "{op}({x:?},{y:?}): {actual:?}, expected {expected:?}, error {} ULP",
            error / unit.into_f64()
        );
    }
    for line in include_str!("data/stable_transforms.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let raw = |i| u64::from_str_radix(fields[i], 16).unwrap();
        if fields[0] == "32" {
            check(
                fields[1],
                f32::from_bits(raw(2) as u32),
                f32::from_bits(raw(3) as u32),
                f32::from_bits(raw(4) as u32),
            );
        } else {
            check(
                fields[1],
                f64::from_bits(raw(2)),
                f64::from_bits(raw(3)),
                f64::from_bits(raw(4)),
            );
        }
    }
}

#[test]
fn independent_mpc_fixtures() {
    fn check<S: RealScalar>(op: &str, z: Complex<S>, w: Complex<S>, expected: Complex<S>) {
        let result = match op {
            "softplus" => softplus(z),
            "sigmoid" => sigmoid(z),
            "log_sigmoid" => log_sigmoid(z),
            "logit" => logit(z),
            "cosm1" => cosm1(z),
            "sinc" => sinc(z),
            "sincpi" => sincpi(z),
            "exprel" => exprel(z),
            "sin_pi" => sin_pi(z),
            "cos_pi" => cos_pi(z),
            "tan_pi" => tan_pi(z),
            "sin_degrees" => sin_degrees(z),
            "cos_degrees" => cos_degrees(z),
            "tan_degrees" => tan_degrees(z),
            "xlogy" => xlogy(z, w),
            "xlog1py" => xlog1py(z, w),
            "powm1" => powm1(z, w),
            _ => panic!("unknown complex fixture {op}"),
        }
        .unwrap_or_else(|e| panic!("{op}({z:?}): {e}; expected {expected:?}"));
        let magnitude = expected
            .re
            .into_f64()
            .abs()
            .max(expected.im.into_f64().abs());
        let growth = match op {
            "sinc" | "cosm1" => 1.0 + z.im.into_f64().abs(),
            "sincpi" | "sin_pi" | "cos_pi" | "tan_pi" => {
                1.0 + core::f64::consts::PI * z.im.into_f64().abs()
            }
            "sin_degrees" | "cos_degrees" | "tan_degrees" => {
                1.0 + (core::f64::consts::PI / 180.0) * z.im.into_f64().abs()
            }
            "exprel" => 1.0 + z.re.into_f64().abs(),
            "powm1" => 1.0 + w.norm().into_f64() * z.ln().norm().into_f64(),
            _ => 1.0,
        };
        let bound = 32.0 * S::EPSILON.into_f64() * magnitude * growth
            + 8.0 * S::ZERO.next_after(S::ONE).into_f64();
        for (a, b) in [(result.re, expected.re), (result.im, expected.im)] {
            assert!(
                (a.into_f64() - b.into_f64()).abs() <= bound,
                "{op}({z:?}): {result:?} != {expected:?}, bound {bound}"
            );
        }
    }
    for line in include_str!("data/stable_complex.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let bits = |i| u64::from_str_radix(f[i], 16).unwrap();
        if f[0] == "32" {
            let z = |i| {
                Complex::new(
                    f32::from_bits(bits(i) as u32),
                    f32::from_bits(bits(i + 1) as u32),
                )
            };
            check(f[1], z(2), z(4), z(6));
        } else {
            let z = |i| Complex::new(f64::from_bits(bits(i)), f64::from_bits(bits(i + 1)));
            check(f[1], z(2), z(4), z(6));
        }
    }
}

#[test]
fn complex_logarithms_and_scaled_sinc() {
    let z = logsumexp(&[Complex::new(1000.0_f64, 0.25), Complex::new(1000.0, 0.25)]).unwrap();
    assert!((z.re - (1000.0 + 2.0_f64.ln())).abs() < 2e-13);
    assert!((z.im - 0.25).abs() < 2e-16);
    assert!(sinc(Complex::new(0.125_f64, 714.0)).is_ok());
    assert_eq!(
        sincpi(Complex::new(3.0_f64, 0.0)).unwrap(),
        Complex::new(0.0, 0.0)
    );
    assert_eq!(
        tan_pi(Complex::new(0.5_f64, 1000.0)).unwrap(),
        Complex::new(0.0, 1.0)
    );
}

#[test]
fn cancellation_is_transactional() {
    let token = CancellationToken::new();
    token.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&token);
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let mut result = [77.0_f64; 4];
    let mut scratch = [0.0; 4];
    assert!(
        scioxide_core::softmax_axis_into(
            NdView::new(&[1.0; 4], layout).unwrap(),
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut result, layout).unwrap(),
            &mut scratch,
            context
        )
        .is_err()
    );
    assert_eq!(result, [77.0; 4]);
}

#[test]
fn weighted_axis_zero_and_complex_broadcast() {
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let weights_layout = StridedLayout::new_broadcast(&[2, 2], &[0, 1]).unwrap();
    let output_layout = StridedLayout::new(&[2], &[1]).unwrap();
    let zero = LogValue::<Complex<f64>, f64>::ZERO;
    let mut out = [zero; 2];
    let mut scratch = [zero; 2];
    scioxide_core::weighted_logsumexp_axis_into(
        NdView::new(&[0.0, 0.0, 1000.0, 1000.0], layout).unwrap(),
        NdView::new(
            &[Complex::new(1.0, 1.0), Complex::new(-1.0, -1.0)],
            weights_layout,
        )
        .unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut out, output_layout).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(out.iter().all(|v| v.log_magnitude().is_none()));
}

struct BrokenExecutor(bool);
impl Executor for BrokenExecutor {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        if !self.0 {
            return Ok(());
        }
        SequentialExecutor.execute(context, plan, operation)?;
        SequentialExecutor.execute(context, plan, operation)
    }
}
#[test]
fn executor_omission_and_replay_are_transactional() {
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    for executor in [BrokenExecutor(false), BrokenExecutor(true)] {
        let mut out = [55.0_f32; 4];
        let mut scratch = [0.0; 4];
        assert!(
            scioxide_core::softmax_axis_into(
                NdView::new(&[1.0; 4], layout).unwrap(),
                1,
                ExecutionPolicy::Sequential,
                NdViewMut::new(&mut out, layout).unwrap(),
                &mut scratch,
                ExecutionContext::new(&executor)
            )
            .is_err()
        );
        assert_eq!(out, [55.0; 4]);
    }
}

#[cfg(feature = "std")]
#[test]
fn executor_normalization_matches_sequential() {
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let layout = StridedLayout::new(&[8, 3], &[3, 1]).unwrap();
    let values: Vec<_> = (0..24).map(|i| (i % 7) as f64 * 0.25).collect();
    let mut sequential = [0.0; 24];
    let mut parallel = [0.0; 24];
    let mut scratch = [0.0; 24];
    scioxide_core::softmax_axis_into(
        NdView::new(&values, layout).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut sequential, layout).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    scioxide_core::softmax_axis_into(
        NdView::new(&values, layout).unwrap(),
        1,
        ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap()),
        NdViewMut::new(&mut parallel, layout).unwrap(),
        &mut scratch,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(parallel, sequential);
}

#[test]
fn complex_axis_and_zero_extent_resources() {
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let output_layout = StridedLayout::new(&[2], &[1]).unwrap();
    let x = [Complex::new(1000.0_f64, 0.25); 4];
    let mut out = [Complex::new(0.0, 0.0); 2];
    let mut scratch = out;
    scioxide_core::logsumexp_axis_into(
        NdView::new(&x, layout).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut out, output_layout).unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    for z in out {
        assert!((z.re - (1000.0 + 2.0_f64.ln())).abs() < 2e-13);
        assert!((z.im - 0.25).abs() < 1e-15);
    }
    let extents = [0, 2, usize::MAX, usize::MAX];
    let strides = [0, 0, 0, 0];
    let layout = StridedLayout::new(&extents, &strides).unwrap();
    scioxide_core::softmax_axis_into(
        NdView::<f64>::new(&[], layout).unwrap(),
        1,
        scioxide_core::ExecutionPolicy::Sequential,
        NdViewMut::new(&mut [], layout).unwrap(),
        &mut [],
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn complex_scaled_trig_retains_subnormal_factors() {
    let z = sin_pi(Complex::new(f64::from_bits(1), 100.0)).unwrap();
    assert!(z.re > 0.0);
    assert!(z.im.is_finite());
}

#[test]
fn complex_powers_reuse_full_range_reconstruction() {
    let value = powm1(Complex::new(0.01_f64, 0.0), Complex::new(f64::MAX, 0.0)).unwrap();
    assert_eq!(value, Complex::new(-1.0, 0.0));
    let phase = powm1(Complex::new(-1.0_f64, 0.0), Complex::new(f64::MAX, 0.0)).unwrap();
    assert!(phase.re.is_finite() && phase.im.is_finite());
}

#[test]
fn destination_overflow_is_numerical_in_real_and_complex_transforms() {
    use scioxide_core::{Complex, NumericalIssue, SciOxideErrorView};
    for error in [
        scioxide_core::exprel(100.0_f32).unwrap_err(),
        scioxide_core::powm1(2.0_f32, 128.0).unwrap_err(),
        scioxide_core::exprel(Complex::new(100.0_f32, 0.0)).unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
    assert!(scioxide_core::exprel(80.0_f32).unwrap().is_finite());
    assert_eq!(scioxide_core::sigmoid(-200.0_f32).unwrap(), 0.0);
}
