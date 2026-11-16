//! Tests of exact identities, operation counts and recursion storage sizes.

use super::*;
use crate::test_counted_ring::{CLONES, Ring};
use crate::test_prime_field::{Fp, measured};
use std::cell::Cell;
type F = Fp<65537>;

fn coefficients(n: usize, seed: u64) -> Vec<F> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = (state * 48271) % 2147483647;
            Fp::<65537>(state % 65536 + 1)
        })
        .collect()
}

fn schoolbook(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = vec![Fp::<65537>(0); a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            result[i + j] = result[i + j] + x * y;
        }
    }
    result
}

#[test]
fn requirements_are_linear_and_monotone() {
    let mut previous = 0;
    for n in (1..=4096).chain((13..=24).map(|power| 1 << power)) {
        let required = ExactPolynomialWorkspace::<F>::workspace_requirements(n)
            .unwrap()
            .coefficient_len();
        assert!(required <= 7 * n, "capacity {n}: {required}");
        assert!(required >= previous);
        previous = required;
    }
    assert_eq!(
        ExactPolynomialWorkspace::<F>::workspace_requirements(1)
            .unwrap()
            .coefficients,
        1
    );
    assert!(ExactPolynomialWorkspace::<F>::workspace_requirements(0).is_err());
    assert!(ExactPolynomialWorkspace::<F>::workspace_requirements(usize::MAX).is_err());
}

#[test]
fn composition_scratch_is_non_decreasing() {
    let mut previous = 0;
    for n in 1..=512 {
        let scratch = composition_scratch(n).unwrap();
        assert!(scratch >= previous, "capacity {n}: {scratch} < {previous}");
        previous = scratch;
    }
}

#[test]
fn balanced_blocked_and_partial_products_match_schoolbook_and_cost_model() {
    let lengths = [1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257];
    for a_len in lengths {
        for b_len in lengths {
            let a = coefficients(a_len, 17);
            let b = coefficients(b_len, 29);
            let reference = schoolbook(&a, &b);
            let mut workspace = ExactPolynomialWorkspace::workspace(reference.len()).unwrap();
            assert_eq!(workspace.data.len(), required_len(reference.len()).unwrap());
            let mut output = vec![Fp::<65537>(0); reference.len()];
            let (_, (products, _)) = measured(|| {
                exact_multiply_into(
                    &a,
                    &b,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap()
            });
            assert_eq!(output, reference, "{a_len} by {b_len}");
            assert_eq!(products, product_cost(a_len, b_len));
        }
    }
    assert!(product_cost(128, 64) < 128 * 64);
    assert!(product_cost(1024, 32) < 1024 * 32);
}

#[test]
fn composition_uses_reported_storage_for_every_small_shape() {
    for outer in 1..=40 {
        for inner in 1..=8 {
            let p = coefficients(outer, 7);
            let q = coefficients(inner, 11);
            let n = (outer - 1) * (inner - 1) + 1;
            let mut workspace = ExactPolynomialWorkspace::workspace(n).unwrap();
            let mut output = vec![Fp::<65537>(0); n];
            exact_compose_into(
                &p,
                &q,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let mut expected = vec![Fp::<65537>(0)];
            for &coefficient in p.iter().rev() {
                expected = schoolbook(&expected, &q);
                expected[0] = expected[0] + coefficient;
            }
            expected.resize(n, Fp::<65537>(0));
            assert_eq!(output, expected, "outer {outer}, inner {inner}");
        }
    }
}

#[test]
fn selected_remainders_beat_classical_products_and_reuse_newton_buffers() {
    let mut selected = 0;
    let balanced = [
        31, 32, 63, 64, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 2048, 4096,
    ]
    .into_iter()
    .map(|q| (q, q));
    for (q, degree) in balanced.chain([(512, 2048), (2048, 512), (513, 1024), (769, 1537)]) {
        let a = ExactPolynomial::new(coefficients(q + degree, 19)).unwrap();
        let mut divisor = coefficients(degree + 1, 37);
        divisor[degree] = Fp::<65537>(1);
        let b = ExactPolynomial::new(divisor).unwrap();
        let mut work = TreeWork::new(remainder_capacity(a.coefficients().len()).unwrap()).unwrap();
        let ((_, expected), (classical, _)) = measured(|| a.div_rem(&b).unwrap());
        let (actual, (products, _)) = measured(|| fast_remainder(&a, &b, &mut work).unwrap());
        assert_eq!(actual, expected, "q = {q}");
        assert!(products <= classical, "q = {q}: {products} > {classical}");
        if use_newton(a.coefficients().len(), b.coefficients().len()) {
            selected += 1;
            assert!(products < classical);
            let pointers = (
                work.inverse.as_ptr(),
                work.reversed.as_ptr(),
                work.error.as_ptr(),
                work.correction.as_ptr(),
            );
            assert_eq!(fast_remainder(&a, &b, &mut work).unwrap(), expected);
            assert_eq!(
                pointers,
                (
                    work.inverse.as_ptr(),
                    work.reversed.as_ptr(),
                    work.error.as_ptr(),
                    work.correction.as_ptr()
                )
            );
        }
    }
    assert!(selected >= 4);
    assert!(use_newton(8192, 4097));
    // A huge quotient against a small divisor must remain classical.
    assert!(!use_newton(8192, 33));
}

#[test]
fn complete_remainder_trees_improve_counts_and_skip_padding() {
    for n in [0, 1, 3, 17, 128, 1024, 1025] {
        let points: Vec<_> = (0..n).map(F::from_usize).collect();
        let tree = ExactPolynomialProductTree::new(&points).unwrap();
        let polynomial = ExactPolynomial::new(coefficients((2 * n).max(1), 43)).unwrap();
        let (expected, (classical, _)) = measured(|| {
            let mut remainders = vec![ExactPolynomial::constant(Fp::<65537>(0)).unwrap()];
            for i in 1..tree.nodes.len() {
                let input = if i == 1 {
                    &polynomial
                } else {
                    &remainders[i / 2]
                };
                remainders.push(if tree.nodes[i].degree() == 0 {
                    ExactPolynomial::constant(Fp::<65537>(0)).unwrap()
                } else {
                    input.div_rem(&tree.nodes[i]).unwrap().1
                });
            }
            (0..n)
                .map(|i| remainders[tree.base + i].coefficients()[0])
                .collect::<Vec<_>>()
        });
        let (actual, (products, _)) = measured(|| tree.evaluate(&polynomial).unwrap());
        assert_eq!(actual, expected);
        assert!(
            products <= classical,
            "points {n}: {products} > {classical}"
        );
        if n >= 1024 {
            assert!(products < classical);
        }
        let constant = ExactPolynomial::constant(Fp::<65537>(7)).unwrap();
        let (_, counts) = measured(|| tree.evaluate(&constant).unwrap());
        assert_eq!(counts, (0, 0), "padding arithmetic at size {n}");
    }
}

#[test]
fn empty_padded_and_duplicate_interpolation_contracts() {
    for n in [0, 1, 3, 17, 129, 1025] {
        let points: Vec<_> = (0..n).map(F::from_usize).collect();
        let tree = ExactPolynomialProductTree::new(&points).unwrap();
        let p = ExactPolynomial::new(if n == 0 {
            vec![Fp::<65537>(0)]
        } else {
            coefficients(n, 47)
        })
        .unwrap();
        assert_eq!(tree.interpolate(&tree.evaluate(&p).unwrap()).unwrap(), p);
        let zero = ExactPolynomial::constant(Fp::<65537>(0)).unwrap();
        assert_eq!(tree.interpolate(&vec![Fp::<65537>(0); n]).unwrap(), zero);
        assert!(tree.interpolate(&vec![Fp::<65537>(0); n + 1]).is_err());
    }
    let duplicate =
        ExactPolynomialProductTree::new(&[Fp::<65537>(1), Fp::<65537>(2), Fp::<65537>(1)]).unwrap();
    assert!(
        duplicate
            .interpolate(&[Fp::<65537>(3), Fp::<65537>(4), Fp::<65537>(3)])
            .is_err()
    );
}

#[test]
fn exact_algebra_operation_counts() {
    let p = |n, seed| ExactPolynomial::new(coefficients(n, seed)).unwrap();
    let rational = crate::ExactRationalFunction::new(p(5, 13), p(4, 17)).unwrap();
    let (_, reciprocal) = measured(|| rational.reciprocal().unwrap());
    let common = p(9, 23);
    let left = common.try_mul(&p(5, 29)).unwrap();
    let right = common.try_mul(&p(7, 31)).unwrap();
    let (_, lcm) = measured(|| left.lcm(&right).unwrap());
    let (_, prs) = measured(|| crate::exact_subresultants(&p(8, 37), &p(6, 41)).unwrap());
    let points: Vec<_> = (0..17).map(F::from_usize).collect();
    let tree = ExactPolynomialProductTree::new(&points).unwrap();
    let polynomial = p(34, 43);
    let (_, tree_counts) = measured(|| tree.evaluate(&polynomial).unwrap());
    // Fixed pre-optimization counts for these fixtures, measured with divisions
    // counted separately from multiplications. No old production path is retained.
    assert_eq!(reciprocal.0, 0);
    assert!(reciprocal.0 < 29);
    assert!(lcm.0 < 489, "LCM: {lcm:?}");
    assert!(prs.0 < 148, "PRS: {prs:?}");
    assert!(tree_counts.0 < 659, "remainders: {tree_counts:?}");
}

#[test]
fn product_only_scratch_is_pinned_and_sufficient() {
    for (n, expected) in [(1, 0), (62, 0), (63, 83), (64, 87), (95, 210), (127, 286)] {
        assert_eq!(product_scratch(n).unwrap(), expected, "length {n}");
        let work = TreeWork::<F>::new(n).unwrap();
        assert_eq!(work.scratch_len, expected);
        assert!(work.scratch.is_empty());
    }
    for n in (1..4096).chain((13..=24).map(|p| 1 << p)) {
        assert!(product_scratch(n).unwrap() <= 4 * n);
    }
    for a_len in [1, 31, 32, 33, 63, 64, 65, 128, 257] {
        for b_len in [1, 31, 32, 33, 63, 64, 65, 128, 257] {
            let a = coefficients(a_len, 19);
            let b = coefficients(b_len, 23);
            let n = a_len + b_len - 1;
            let mut out = vec![Fp::<65537>(0); n];
            let mut scratch = vec![Fp::<65537>(0); product_scratch(n).unwrap()];
            karatsuba(
                &a,
                &b,
                &mut out,
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(out, schoolbook(&a, &b));
        }
    }
}

#[test]
fn shared_polynomial_powers_match_independent_products() {
    for len in 1..=6 {
        let polynomial = ExactPolynomial::new(coefficients(len, 53)).unwrap();
        let mut expected = vec![Fp::<65537>(1)];
        for exponent in 0..=17 {
            assert_eq!(
                power(&polynomial, exponent).unwrap().coefficients(),
                expected
            );
            expected = schoolbook(&expected, polynomial.coefficients());
        }
    }
    let zero = ExactPolynomial::constant(Fp::<65537>(0)).unwrap();
    assert_eq!(power(&zero, 0).unwrap().coefficients(), [Fp::<65537>(1)]);
    assert_eq!(power(&zero, usize::MAX).unwrap(), zero);
    let one = ExactPolynomial::constant(Fp::<65537>(1)).unwrap();
    assert_eq!(power(&one, usize::MAX).unwrap(), one);
    let x = ExactPolynomial::new(vec![Fp::<65537>(0), Fp::<65537>(1)]).unwrap();
    assert!(matches!(
        power(&x, usize::MAX).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn remainder_only_matches_division_including_monic_and_zero_cases() {
    let zero = ExactPolynomial::constant(Fp::<65537>(0)).unwrap();
    for length in 1..=12 {
        let a = ExactPolynomial::new(coefficients(length, 59)).unwrap();
        assert!(a.remainder(&zero).is_err());
        for divisor_length in 1..=14 {
            let b = ExactPolynomial::new(coefficients(divisor_length, 61)).unwrap();
            for divisor in [&b, &b.monic().unwrap()] {
                assert_eq!(a.remainder(divisor).unwrap(), a.div_rem(divisor).unwrap().1);
                assert_eq!(zero.remainder(divisor).unwrap(), zero);
            }
        }
    }
}

#[test]
fn prepared_product_publication_does_not_clone_coefficients() {
    for len in [1, 31, 32, 33, 64] {
        let a: Vec<_> = coefficients(len, 67).into_iter().map(Ring).collect();
        let mut workspace = ExactPolynomialWorkspace::<Ring>::workspace(2 * len - 1).unwrap();
        let mut expected = vec![Ring::zero(); 2 * len - 1];
        let mut actual = expected.clone();
        let mut scratch = vec![Ring::zero(); product_scratch(expected.len()).unwrap()];
        CLONES.with(|n| n.set(0));
        karatsuba(
            &a,
            &a,
            &mut expected,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let kernel_clones = CLONES.with(Cell::get);
        for _ in 0..2 {
            CLONES.with(|n| n.set(0));
            exact_multiply_into(
                &a,
                &a,
                &mut actual,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(CLONES.with(Cell::get), kernel_clones);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn exact_interpolation_has_independent_coefficients() {
    use crate::test_field::Q;
    let q = |v| Q(v, 1);
    let polynomial = crate::ExactPolynomialProductTree::new(&[q(0), q(1), q(2)])
        .unwrap()
        .interpolate(&[q(1), q(3), q(7)])
        .unwrap();
    assert_eq!(polynomial.coefficients(), &[q(1), q(1), q(1)]);
}
