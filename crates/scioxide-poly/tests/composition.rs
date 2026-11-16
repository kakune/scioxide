//! Exact-rational composition references, including gradual underflow tails.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, ExecutionContext, NumericScalar};

use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
use std::collections::BTreeMap;

fn coefficients(family: &str, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| match family {
            "monomial" => f64::from(k == n - 1),
            "signed" if k % 2 == 1 => -((k % 7 + 1) as f64) / 8.0,
            _ => (k % 7 + 1) as f64 / 8.0,
        })
        .collect()
}

#[test]
fn exact_rational_references_pin_normal_and_underflowing_coefficients() {
    let mut cases = BTreeMap::<(&str, usize), Vec<(f64, f64)>>::new();
    for row in include_str!("data/composition.csv").lines().skip(1) {
        let fields: Vec<_> = row.split(',').collect();
        let n = fields[1].parse().unwrap();
        let expected = cases.entry((fields[0], n)).or_default();
        assert_eq!(fields[2].parse::<usize>().unwrap(), expected.len());
        expected.push((fields[3].parse().unwrap(), fields[4].parse().unwrap()));
    }
    assert_eq!(cases.len(), 9);
    let mut subnormal = 0;
    let mut rounded_zero = 0;
    let mut tiny_tail = 0;
    for ((family, n), expected) in cases {
        let request = PolynomialOperationRequest::Compose {
            outer_len: n,
            inner_len: 3,
        };
        let plan = PolynomialOperationPlan::<f64>::new(&[request]).unwrap();
        let mut workspace = plan.workspace(&[request]).unwrap();
        let p = coefficients(family, n);
        let mut output = vec![f64::NAN; 2 * n - 1];
        plan.compose_into(
            &p,
            &[0.25, 0.5, 0.25],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        let mut maximum_error = 0.0_f64;
        let mut maximum_majorant = 0.0_f64;
        for (j, (&actual, &(reference, majorant))) in output.iter().zip(&expected).enumerate() {
            let error = (actual - reference).abs();
            // A componentwise bound is essential here: the largest coefficient
            // must not set an absolute floor for the underflowing tail.
            let tolerance = 32.0 * f64::EPSILON * majorant + 2.0 * f64::from_bits(1);
            assert!(
                error <= tolerance,
                "{family}/{n}/{j}: {actual:e}, expected {reference:e}, error {error:e}, bound {tolerance:e}"
            );
            maximum_error = maximum_error.max(error);
            maximum_majorant = maximum_majorant.max(majorant);
            subnormal += usize::from(reference.is_subnormal());
            rounded_zero += usize::from(family == "monomial" && reference == 0.0);
            tiny_tail += usize::from(reference != 0.0 && reference.abs() < 1e-290);
        }
        assert!(maximum_error <= 16.0 * f64::EPSILON * maximum_majorant);
    }
    assert!(subnormal > 0);
    assert!(rounded_zero > 0);
    assert!(tiny_tail > 200);
}

#[test]
fn complex_binary_blocks_retain_irregular_lengths_and_explicit_zero_tails() {
    for n in [512, 513, 1024] {
        let plan =
            PolynomialOperationPlan::<Complex<f64>>::new(&[PolynomialOperationRequest::Compose {
                outer_len: n,
                inner_len: 3,
            }])
            .unwrap();
        let mut workspace = plan.workspace(plan.requests()).unwrap();
        let mut p = vec![Complex::new(0.0, 0.0); n];
        p[0] = Complex::new(0.5, -0.25);
        p[n - 1] = Complex::new(0.25, 0.5);
        let mut out = vec![Complex::new(99.0, 99.0); 2 * n - 1];
        plan.compose_into(
            &p,
            &[
                Complex::new(0.0, 0.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.0, 0.0),
            ],
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        let expected = p[n - 1] * 0.5_f64.powi((n - 1) as i32);
        // Sparse operands are FFT-eligible, with normwise rather than exact-zero
        // accuracy. Compare the small expanded term on its own scale.
        let tolerance = 4096.0 * f64::EPSILON * expected.abs() + f64::from_bits(1);
        for (j, &actual) in out.iter().enumerate() {
            let reference = if j == 0 {
                p[0]
            } else if j == n - 1 {
                expected
            } else {
                Complex::new(0.0, 0.0)
            };
            assert!(
                (actual - reference).abs() <= tolerance,
                "{n}/{j}: {actual:?}, expected {reference:?}"
            );
        }
    }
}
