//! Independent exact identities and sparse/dense layout coverage.
#![allow(clippy::unwrap_used)]
#[cfg(feature = "std")]
use core::num::NonZeroUsize;
use scioxide_core::*;
use scioxide_poly::*;
#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;
fn m(e: &[usize]) -> Vec<usize> {
    e.to_vec()
}
fn p() -> ApproxSparsePolynomial<f64> {
    ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::GradedReverseLexicographic,
        vec![(m(&[2, 0]), 2.), (m(&[1, 1]), 3.), (m(&[0, 0]), -1.)],
    )
    .unwrap()
}
#[test]
fn canonical_exact_arithmetic() {
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        let a = ExactSparsePolynomial::from_terms(
            2,
            order,
            vec![
                (m(&[1, 0]), Q(2, 1)),
                (m(&[0, 1]), Q(3, 1)),
                (m(&[1, 0]), Q(-1, 1)),
            ],
        )
        .unwrap();
        assert_eq!(a.terms().len(), 2);
        assert!(a.try_sub(&a).unwrap().is_zero());
        assert_eq!(a.evaluate(&[Q(2, 1), Q(1, 3)]).unwrap(), Q(3, 1));
        assert_eq!(
            a.try_mul(&a)
                .unwrap()
                .evaluate(&[Q(2, 1), Q(1, 3)])
                .unwrap(),
            Q(9, 1)
        );
        assert_eq!(
            a.derivative(1)
                .unwrap()
                .evaluate(&[Q(0, 1), Q(0, 1)])
                .unwrap(),
            Q(3, 1)
        );
        assert_eq!(
            a.permute_variables(&[1, 0])
                .unwrap()
                .evaluate(&[Q(1, 3), Q(2, 1)])
                .unwrap(),
            Q(3, 1)
        );
    }
}
#[test]
fn all_numerical_domains() {
    fn check<C: NumericScalar>() {
        let c = |n| C::from_real(C::Real::try_from_f64_lossy(n).unwrap());
        let a = ApproxSparsePolynomial::from_terms(
            2,
            MonomialOrder::Lexicographic,
            vec![
                (m(&[2, 0]), c(2.)),
                (m(&[1, 1]), c(3.)),
                (m(&[0, 0]), c(-1.)),
            ],
        )
        .unwrap();
        assert_eq!(a.evaluate(&[c(2.), c(3.)]).unwrap(), c(25.));
        assert_eq!(
            a.derivative(0).unwrap().evaluate(&[c(2.), c(3.)]).unwrap(),
            c(17.)
        );
        assert_eq!(
            a.try_mul(&a).unwrap().evaluate(&[c(2.), c(3.)]).unwrap(),
            c(625.)
        );
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
    let a = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![(m(&[2]), Complex::new(1., 1.))],
    )
    .unwrap();
    assert_eq!(
        a.evaluate(&[Complex::new(0., 1.)]).unwrap(),
        Complex::new(-1., -1.)
    );
}
#[test]
fn substitution_limits_and_simultaneous_semantics() {
    let a = p();
    let r = |e| ApproxSparsePolynomial::from_terms(2, a.order(), vec![(m(e), 1.)]).unwrap();
    let replacements = [r(&[0, 1]), r(&[1, 0])];
    let limits = SparseExpansionLimits::new(8);
    let swapped = a.substitute_with_options(&replacements, 2, limits).unwrap();
    // a(y,x) = 2y² + 3xy - 1, expanded independently of permutation.
    let expected = ApproxSparsePolynomial::from_terms(
        2,
        a.order(),
        vec![(m(&[0, 2]), 2.), (m(&[1, 1]), 3.), (m(&[0, 0]), -1.)],
    )
    .unwrap();
    assert_eq!(swapped, expected);
    assert_eq!(swapped.evaluate(&[2., 3.]).unwrap(), 35.);
    let mut output = a.clone();
    a.substitute_into(
        &replacements,
        2,
        limits,
        &mut output,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, swapped);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        a.substitute_into(
            &replacements,
            2,
            limits,
            &mut output,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, swapped);
    let error = a
        .substitute_with_options(&replacements, 2, SparseExpansionLimits::new(1))
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ResourceLimit { .. }
    ));
    assert_eq!(a, p());
}

#[test]
fn substitution_rejects_shapes_orders_and_zero_limits_atomically() {
    let a = p();
    let valid = ApproxSparsePolynomial::from_terms(2, a.order(), vec![(m(&[1, 0]), 1.)]).unwrap();
    let wrong_dimension =
        ApproxSparsePolynomial::from_terms(1, a.order(), vec![(m(&[1]), 1.)]).unwrap();
    let wrong_order =
        ApproxSparsePolynomial::from_terms(2, MonomialOrder::Lexicographic, vec![(m(&[1, 0]), 1.)])
            .unwrap();
    for (replacements, target, limits, kind) in [
        (
            vec![valid.clone()],
            2,
            SparseExpansionLimits::new(32),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            vec![wrong_dimension.clone(), wrong_dimension],
            2,
            SparseExpansionLimits::new(32),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            vec![wrong_order.clone(), wrong_order],
            2,
            SparseExpansionLimits::new(32),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            vec![valid.clone(), valid],
            2,
            SparseExpansionLimits::new(0),
            SciOxideErrorKind::ResourceLimit,
        ),
    ] {
        let mut output = a.clone();
        assert_eq!(
            a.substitute_into(
                &replacements,
                target,
                limits,
                &mut output,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            kind
        );
        assert_eq!(output, a);
    }
    // Exact substitution follows the same shape and transaction contract.
    let exact = ExactSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![(m(&[2]), Q(1, 1))],
    )
    .unwrap();
    let replacement =
        ExactSparsePolynomial::from_terms(1, exact.order(), vec![(m(&[1]), Q(1, 1))]).unwrap();
    let mut output = exact.clone();
    assert_eq!(
        exact
            .substitute_into(
                &[replacement],
                1,
                SparseExpansionLimits::new(0),
                &mut output,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(output, exact);
}
#[test]
fn tensor_batch_and_transaction() {
    let a = p();
    let mut dense = [99.; 9];
    let layout = StridedLayout::new(&[3, 3], &[3, 1]).unwrap();
    a.to_tensor_into(
        NdViewMut::new(&mut dense, layout).unwrap(),
        &[1, 0],
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let b = ApproxSparsePolynomial::from_tensor(
        NdView::new(&dense, layout).unwrap(),
        &[1, 0],
        a.order(),
    )
    .unwrap();
    assert_eq!(a, b);
    let points = [2., 3., 1., 4.];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let mut output = [77.; 2];
    let mut scratch = vec![
        0.;
        a.evaluation_plan(Default::default())
            .unwrap()
            .workspace_requirements(2)
            .unwrap()
            .scratch_len()
    ];
    a.evaluation_plan(scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap()
        .evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut output,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [25., 13.]);
    let invalid = [2., 3., f64::NAN, 4.];
    assert!(
        a.evaluation_plan(scioxide_poly::PolynomialExecutionOptions::new())
            .unwrap()
            .evaluate_batch_into(
                NdView::new(&invalid, layout).unwrap(),
                &mut output,
                &mut scratch,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [25., 13.]);
}
#[test]
fn overflow_and_zero_dimensions() {
    assert!(matches!(
        (Monomial::new(&[usize::MAX, 1])).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
    let large = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        [([usize::MAX], 1.0_f64)],
    )
    .unwrap();
    let linear =
        ApproxSparsePolynomial::from_terms(1, MonomialOrder::Lexicographic, [([1], 1.0_f64)])
            .unwrap();
    assert!(large.try_mul(&linear).is_err());
    assert!(p().permute_variables(&[0, 0]).is_err());
    assert!(p().derivative(2).is_err());
    let z = ApproxSparsePolynomial::<f64>::from_terms(
        0,
        MonomialOrder::Lexicographic,
        Vec::<([usize; 0], f64)>::new(),
    )
    .unwrap();
    assert_eq!(z.evaluate(&[]).unwrap(), 0.);
    let constant = ApproxSparsePolynomial::<f64>::from_terms(
        0,
        MonomialOrder::Lexicographic,
        vec![(m(&[]), 7.)],
    )
    .unwrap();
    assert_eq!(constant.evaluate(&[]).unwrap(), 7.);
    assert!(p().try_add(&constant).is_err());
}

#[test]
#[cfg(feature = "std")]
fn executor_parity_stops_and_arithmetic_transaction() {
    let workers = NonZeroUsize::new(3).unwrap();
    let executor = ThreadExecutor::new(workers);
    let a = p();
    let points = [2., 3., 1., 4., -2., 1., 0., 0., 4., -1.];
    let shape = [5, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let mut seq = [0.; 5];
    let mut par = [0.; 5];
    let plan = a
        .evaluation_plan(
            scioxide_poly::PolynomialExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
    let mut scratch = vec![0.; plan.workspace_requirements(5).unwrap().scratch_len()];
    plan.evaluate_batch_into(
        NdView::new(&points, layout).unwrap(),
        &mut seq,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    plan.evaluate_batch_into(
        NdView::new(&points, layout).unwrap(),
        &mut par,
        &mut scratch,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    assert_eq!(seq, par);
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(
        plan.evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut par,
            &mut scratch,
            ExecutionContext::new(&executor).with_cancellation(&cancel)
        )
        .is_err()
    );
    assert_eq!(par, seq);
    let overflow = [2., 3., f64::MAX, 1., -2., 1., 0., 0., 4., -1.];
    assert!(
        plan.evaluate_batch_into(
            NdView::new(&overflow, layout).unwrap(),
            &mut par,
            &mut scratch,
            ExecutionContext::new(&executor)
        )
        .is_err()
    );
    assert_eq!(par, seq);
}
#[test]
fn empty_strided_and_range_safe_batches() {
    let a = p();
    let layout = StridedLayout::new(&[0, 2], &[2, 1]).unwrap();
    a.evaluation_plan(scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap()
        .evaluate_batch_into(
            NdView::new(&[], layout).unwrap(),
            &mut [],
            &mut [],
            ExecutionContext::sequential(),
        )
        .unwrap();
    let points = [2., 1., 3., 4.];
    let layout = StridedLayout::new(&[2, 2], &[1, 2]).unwrap();
    let mut output = [0.; 2];
    a.evaluation_plan(scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap()
        .evaluate_batch_into(
            NdView::new(&points, layout).unwrap(),
            &mut output,
            &mut [0.; 6],
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [25., 13.]);
    let p = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![(m(&[2]), 1e-300)],
    )
    .unwrap();
    assert!((p.evaluate(&[1e200]).unwrap() / 1e100 - 1.).abs() < 1e-14);
    let exponents = [1, 1];
    let coefficients = [1., 2.];
    assert!(
        SparseEvaluationPlan::new(
            1,
            MonomialOrder::Lexicographic,
            &exponents,
            &coefficients,
            scioxide_poly::PolynomialExecutionOptions::new()
        )
        .is_err()
    );
}

#[test]
fn exact_expanding_substitution_and_chain_rule() {
    let order = MonomialOrder::Lexicographic;
    let make = |terms: Vec<(&[usize], i128)>| {
        ExactSparsePolynomial::from_terms(2, order, terms.into_iter().map(|(e, c)| (m(e), Q(c, 1))))
            .unwrap()
    };
    let a = make(vec![(&[2, 0], 1), (&[1, 1], 1), (&[0, 0], 2)]);
    let x = make(vec![(&[1, 0], 1), (&[0, 0], 1)]);
    let y = make(vec![(&[0, 1], 1), (&[1, 0], -1)]);
    let replacements = [x, y];
    let limits = SparseExpansionLimits::new(32);
    let composed = a.substitute_with_options(&replacements, 2, limits).unwrap();
    let dx = a
        .derivative(0)
        .unwrap()
        .substitute_with_options(&replacements, 2, limits)
        .unwrap();
    let dy = a
        .derivative(1)
        .unwrap()
        .substitute_with_options(&replacements, 2, limits)
        .unwrap();
    assert_eq!(composed.derivative(0).unwrap(), dx.try_sub(&dy).unwrap());
    for i in -3..=3 {
        let point = [Q(i, 2), Q(3, 4)];
        let replaced = [
            replacements[0].evaluate(&point).unwrap(),
            replacements[1].evaluate(&point).unwrap(),
        ];
        assert_eq!(
            composed.evaluate(&point).unwrap(),
            a.evaluate(&replaced).unwrap()
        );
    }
}
#[test]
fn checked_failures_leave_dense_output_unchanged() {
    let a = p();
    let mut dense = [19.; 4];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    assert!(
        a.to_tensor_into(
            NdViewMut::new(&mut dense, layout).unwrap(),
            &[0, 1],
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(dense, [19.; 4]);
    let make =
        |c| ApproxSparsePolynomial::from_terms(1, MonomialOrder::Lexicographic, vec![(m(&[0]), c)]);
    assert!(matches!(
        (make(f64::NAN)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(make(f64::MAX).unwrap().try_mul(&make(2.).unwrap()).is_err());
    let tiny = make(f64::MIN_POSITIVE).unwrap();
    assert!(!tiny.is_zero());
    assert!(a.evaluate(&[1.]).is_err_and(|error| matches!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
            | scioxide_core::SciOxideErrorKind::DimensionMismatch
    )));
}

#[test]
fn exact_borrowed_batches() {
    let exponents = [1, 1];
    let coefficients = [Q(3, 2)];
    let plan =
        ExactSparseView::new(2, MonomialOrder::Lexicographic, &exponents, &coefficients).unwrap();
    let points = [Q(2, 1), Q(1, 3), Q(-1, 1), Q(2, 1)];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let mut output = [Q(0, 1); 2];
    let mut scratch = [Q(0, 1); 2];
    plan.evaluate_batch_into(
        NdView::new(&points, layout).unwrap(),
        &mut output,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [Q(1, 1), Q(-3, 1)]);
}

fn assert_canonical<C>(terms: SparseTerms<'_, C>, order: MonomialOrder) {
    let terms: Vec<_> = terms.iter().collect();
    assert!(terms.windows(2).all(|pair| {
        order.compare_monomials(&pair[0].0, &pair[1].0).unwrap() == core::cmp::Ordering::Less
    }));
}

#[test]
fn large_sparse_operations_remain_canonical_in_every_order() {
    let orders = [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ];
    for order in orders {
        let mut terms: Vec<_> = (0..256)
            .rev()
            .map(|i| (m(&[i % 16, i / 16]), ((i % 7) + 1) as f64))
            .collect();
        terms.extend([
            (m(&[20, 0]), 3.),
            (m(&[1, 0]), -2.),
            (m(&[20, 0]), -3.),
            (m(&[0, 0]), 0.),
        ]);
        let a = ApproxSparsePolynomial::from_terms(2, order, terms.clone()).unwrap();
        let exact = ExactSparsePolynomial::from_terms(
            2,
            order,
            terms.into_iter().map(|(m, c)| (m, Q(c as i128, 1))),
        )
        .unwrap();
        assert_eq!(a.terms().len(), 255);
        assert_eq!(exact.terms().len(), 255);
        for result in [
            a.try_add(&a).unwrap(),
            a.derivative(0).unwrap(),
            a.derivative(1).unwrap(),
            a.permute_variables(&[1, 0]).unwrap(),
        ] {
            assert_canonical(result.terms(), order);
        }
        for result in [
            exact.try_add(&exact).unwrap(),
            exact.derivative(0).unwrap(),
            exact.derivative(1).unwrap(),
        ] {
            assert_canonical(result.terms(), order);
        }
        assert!(a.try_sub(&a).unwrap().is_zero());
        assert!(exact.try_sub(&exact).unwrap().is_zero());
        for target in orders {
            assert_canonical(a.with_order(target).unwrap().terms(), target);
            assert_canonical(exact.with_order(target).unwrap().terms(), target);
        }
        let small = ApproxSparsePolynomial::from_terms(
            2,
            order,
            a.terms().iter().take(16).map(|(m, c)| (m, *c)),
        )
        .unwrap();
        let small_exact = ExactSparsePolynomial::from_terms(
            2,
            order,
            exact.terms().iter().take(16).map(|(m, c)| (m, *c)),
        )
        .unwrap();
        assert_canonical(small.try_mul(&small).unwrap().terms(), order);
        assert_canonical(small_exact.try_mul(&small_exact).unwrap().terms(), order);
        let identity = [vec![1, 0], vec![0, 1]]
            .map(|e| ApproxSparsePolynomial::from_terms(2, order, vec![(e, 1.)]).unwrap());
        let substituted = a
            .substitute_with_options(&identity, 2, SparseExpansionLimits::new(256))
            .unwrap();
        assert_canonical(substituted.terms(), order);
        assert_eq!(substituted, a);
        let identity = [vec![1, 0], vec![0, 1]]
            .map(|e| ExactSparsePolynomial::from_terms(2, order, vec![(e, Q(1, 1))]).unwrap());
        let substituted = exact
            .substitute_with_options(&identity, 2, SparseExpansionLimits::new(256))
            .unwrap();
        assert_canonical(substituted.terms(), order);
        assert_eq!(substituted, exact);
    }
}

#[test]
fn duplicate_folds_keep_original_coefficient_order() {
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        let make = |coefficients: &[f64]| {
            ApproxSparsePolynomial::from_terms(
                1,
                order,
                coefficients.iter().enumerate().map(|(i, &c)| (m(&[i]), c)),
            )
            .unwrap()
        };
        let p = ApproxSparsePolynomial::from_terms(
            1,
            order,
            vec![
                (m(&[1]), 1e16),
                (m(&[0]), 2.),
                (m(&[1]), -1e16),
                (m(&[1]), 1.),
            ],
        )
        .unwrap();
        assert_eq!(p.coefficients()[1], 1.);
        // The smaller right operand supplies heap rows; ties still follow left/right indices.
        let product = make(&[1e16, -1e16, 1., 2.])
            .try_mul(&make(&[1., 1., 1.]))
            .unwrap();
        assert_eq!(
            *product
                .terms()
                .iter()
                .find(|(m, _)| m.exponents() == [2])
                .unwrap()
                .1,
            1.
        );
    }
}

#[test]
fn product_limits_count_folded_terms_and_bound_intermediate_squares() {
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        let make = |terms: Vec<(&[usize], f64)>| {
            ApproxSparsePolynomial::from_terms(2, order, terms.into_iter().map(|(e, c)| (m(e), c)))
                .unwrap()
        };
        let xy = make(vec![(&[1, 1], 1.)]);
        let plus = make(vec![(&[0, 0], 1.), (&[1, 0], 1.)]);
        let minus = make(vec![(&[0, 0], 1.), (&[1, 0], -1.)]);
        let result = xy
            .substitute_with_options(&[plus.clone(), minus], 2, SparseExpansionLimits::new(2))
            .unwrap();
        assert_eq!(result, make(vec![(&[0, 0], 1.), (&[2, 0], -1.)]));
        let square = make(vec![(&[2, 0], 1.)]);
        assert!(matches!(
            square
                .substitute_with_options(
                    &[plus.clone(), plus.clone()],
                    2,
                    SparseExpansionLimits::new(2)
                )
                .unwrap_err()
                .view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        assert_eq!(
            square
                .substitute_with_options(&[plus.clone(), plus], 2, SparseExpansionLimits::new(3))
                .unwrap()
                .terms()
                .len(),
            3
        );
    }
}

#[test]
fn heap_products_match_independent_integer_convolution() {
    use std::collections::BTreeMap;
    let left: Vec<_> = (0..80)
        .map(|i| (vec![i % 10, i / 10], (i % 5) as i128 - 2))
        .collect();
    let right: Vec<_> = (0..35)
        .map(|i| (vec![i % 7, i / 7], (i % 7) as i128 - 3))
        .collect();
    let mut expected = BTreeMap::new();
    for (a, x) in &left {
        for (b, y) in &right {
            *expected.entry(vec![a[0] + b[0], a[1] + b[1]]).or_insert(0) += x * y;
        }
    }
    expected.retain(|_, c| *c != 0);
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        let make = |terms: &[(Vec<usize>, i128)]| {
            ExactSparsePolynomial::from_terms(
                2,
                order,
                terms.iter().rev().map(|(e, c)| (m(e), Q(*c, 1))),
            )
            .unwrap()
        };
        let a = make(&left);
        let b = make(&right);
        for product in [a.try_mul(&b).unwrap(), b.try_mul(&a).unwrap()] {
            assert_canonical(product.terms(), order);
            let actual: BTreeMap<_, _> = product
                .terms()
                .iter()
                .map(|(m, c)| (m.exponents().to_vec(), c.0 / c.1))
                .collect();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn shared_power_evaluation_matches_independent_repeated_products() {
    fn check<C: NumericScalar>() {
        let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
        let point = [
            C::from_components(r(0.5), if C::IS_REAL { r(0.) } else { r(0.25) }),
            C::from_components(r(0.25), if C::IS_REAL { r(0.) } else { r(-0.125) }),
        ];
        let terms: Vec<_> = (0..128)
            .map(|i| {
                (
                    m(&[i % 16, i / 16]),
                    C::from_real(r((i % 16 + 2 * (i / 16) + 1) as f64)),
                )
            })
            .collect();
        let expected = terms.iter().fold(C::ZERO, |sum, (m, coefficient)| {
            let value = m.iter().zip(point).fold(*coefficient, |value, (&e, x)| {
                (0..e).fold(value, |v, _| v * x)
            });
            sum + value
        });
        for order in [
            MonomialOrder::Lexicographic,
            MonomialOrder::GradedLexicographic,
            MonomialOrder::GradedReverseLexicographic,
        ] {
            let p = ApproxSparsePolynomial::from_terms(2, order, terms.clone()).unwrap();
            let plan = p.evaluation_plan(Default::default()).unwrap();
            let mut scratch = vec![C::ZERO; plan.workspace_requirements(1).unwrap().scratch_len()];
            let actual = plan.evaluate(&point, &mut scratch).unwrap();
            assert!((actual - expected).abs() <= r(128.) * C::Real::EPSILON * expected.abs());
            assert_eq!(p.evaluate(&point).unwrap(), actual);
        }
        // Powers overflow native range, but multiplying by the tiny coefficient
        // leaves a representable final result in both scalar widths.
        let p = ApproxSparsePolynomial::from_terms(
            1,
            MonomialOrder::Lexicographic,
            vec![(m(&[2]), C::from_real(C::Real::MIN_POSITIVE))],
        )
        .unwrap();
        let expected =
            C::from_real((C::Real::MIN_POSITIVE * C::Real::MAX) * (C::Real::MAX / r(16.)));
        let actual = p.evaluate(&[C::from_real(C::Real::MAX / r(4.))]).unwrap();
        assert!((actual - expected).abs() <= r(16.) * C::Real::EPSILON * expected.abs());
        assert!(p.evaluate(&[C::from_real(C::Real::MAX)]).is_err());
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

#[test]
fn sparse_scalar_scratch_boundaries_and_huge_exponents() {
    let p = ApproxSparsePolynomial::from_terms(
        1,
        MonomialOrder::Lexicographic,
        vec![(m(&[usize::MAX]), 3.0_f64)],
    )
    .unwrap();
    let plan = p.evaluation_plan(Default::default()).unwrap();
    let required = plan.workspace_requirements(1).unwrap().scratch_len();
    assert_eq!(required, usize::BITS as usize + 1);
    let mut scratch = vec![0.; required + 3];
    assert!(matches!(
        plan.evaluate(&[1.], &mut scratch[..required - 1])
            .unwrap_err()
            .view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(plan.evaluate(&[1.], &mut scratch[..required]).unwrap(), 3.);
    assert_eq!(plan.evaluate(&[-1.], &mut scratch).unwrap(), -3.);
    assert_eq!(plan.evaluate(&[0.], &mut scratch).unwrap(), 0.);
    assert!(p.evaluate(&[f64::NAN]).is_err());
    assert!(plan.workspace_requirements(usize::MAX).is_err());
}

#[test]
fn complex_coefficient_multiplication_retains_exceptional_component_range() {
    let make = |c| {
        ApproxSparsePolynomial::from_terms(0, MonomialOrder::Lexicographic, vec![(m(&[]), c)])
            .unwrap()
    };
    let a = Complex::new(f64::MAX, f64::MAX * 0.3);
    let b = Complex::new(1.1, 0.5);
    assert!(!(a * b).re.is_finite());
    let product = make(a).try_mul(&make(b)).unwrap();
    let coefficient = product.coefficients()[0];
    assert!((coefficient.re / f64::MAX - 0.95).abs() < 1e-14);
    assert!((coefficient.im / f64::MAX - 0.83).abs() < 1e-14);
}

#[test]
fn flat_storage_validates_shape_degrees_and_canonicalizes_stably() {
    let order = MonomialOrder::GradedLexicographic;
    let p = ApproxSparsePolynomial::new(
        2,
        order,
        vec![1, 0, 0, 1, 1, 0, 1, 0],
        vec![1e16_f64, 2., -1e16, 1.],
    )
    .unwrap();
    assert_eq!(p.exponents(), &[0, 1, 1, 0]);
    assert_eq!(p.coefficients(), &[2., 1.]);
    assert_eq!(
        p.terms()
            .iter()
            .map(|(m, _)| m.degree())
            .collect::<Vec<_>>(),
        [1, 1]
    );
    for (dimension, exponents, coefficients, kind) in [
        (2, vec![1], vec![1.], SciOxideErrorKind::DimensionMismatch),
        (0, vec![1], vec![1.], SciOxideErrorKind::DimensionMismatch),
        (
            2,
            vec![usize::MAX, 1],
            vec![1.],
            SciOxideErrorKind::SizeOverflow,
        ),
        (1, vec![0], vec![f64::NAN], SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(
            ApproxSparsePolynomial::new(dimension, order, exponents, coefficients)
                .unwrap_err()
                .kind(),
            kind
        );
    }
    let p = ExactSparsePolynomial::new(0, order, vec![], vec![Q(2, 1), Q(-2, 1), Q(3, 1)]).unwrap();
    assert_eq!(p.coefficients(), &[Q(3, 1)]);
    assert_eq!(p.terms().iter().next().unwrap().0.exponents(), &[]);
}

#[test]
fn substitution_checks_every_canonical_source_prefix_before_publication() {
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        let source = ApproxSparsePolynomial::from_terms(
            3,
            order,
            [([1, 0, 0], 1.0_f64), ([0, 1, 0], 1.), ([0, 0, 1], 1.)],
        )
        .unwrap();
        let make = |exponent, coefficient| {
            ApproxSparsePolynomial::new(1, order, vec![exponent], vec![coefficient]).unwrap()
        };
        // Source order is x2, x1, x0. The prefix 1+x exceeds the cap even
        // though the last source term cancels x from the final result.
        let replacements = [make(1, -1.), make(1, 1.), make(0, 1.)];
        let mut output = make(2, 7.);
        let original = output.clone();
        let error = source
            .substitute_into(
                &replacements,
                1,
                SparseExpansionLimits::new(1),
                &mut output,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
        assert_eq!(output, original);
        assert_eq!(
            source
                .substitute_with_options(&replacements, 1, SparseExpansionLimits::new(2))
                .unwrap(),
            make(0, 1.)
        );
        let cancelling = [make(1, 1.), make(0, -1.), make(0, 1.)];
        assert_eq!(
            source
                .substitute_with_options(&cancelling, 1, SparseExpansionLimits::new(1))
                .unwrap(),
            make(1, 1.)
        );

        let exact =
            ExactSparsePolynomial::new(3, order, source.exponents().to_vec(), vec![Q(1, 1); 3])
                .unwrap();
        let exact_replacements = replacements.map(|p| {
            ExactSparsePolynomial::new(
                1,
                order,
                p.exponents().to_vec(),
                p.coefficients().iter().map(|&c| Q(c as i128, 1)).collect(),
            )
            .unwrap()
        });
        assert_eq!(
            exact
                .substitute_with_options(&exact_replacements, 1, SparseExpansionLimits::new(1))
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            exact
                .substitute_with_options(&exact_replacements, 1, SparseExpansionLimits::new(2))
                .unwrap()
                .coefficients(),
            &[Q(1, 1)]
        );
    }
}

#[test]
fn single_active_axis_products_match_independent_integer_convolution() {
    let left = [1_i128, -2, 3, -4, 5, -6, 7, -8];
    let right = [3_i128, 1, -2, 4, 2];
    let mut expected = [0_i128; 12];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            expected[i + j] += a * b;
        }
    }
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        for axis in 0..3 {
            let make = |coefficients: &[i128]| {
                ExactSparsePolynomial::from_terms(
                    3,
                    order,
                    coefficients.iter().enumerate().map(|(i, &c)| {
                        let mut e = [0; 3];
                        e[axis] = i;
                        (e, Q(c, 1))
                    }),
                )
                .unwrap()
            };
            let (a, b) = (make(&left), make(&right));
            for p in [a.try_mul(&b).unwrap(), b.try_mul(&a).unwrap()] {
                assert_canonical(p.terms(), order);
                let mut actual = [0_i128; 12];
                for (m, c) in p.terms() {
                    assert!(
                        m.exponents()
                            .iter()
                            .enumerate()
                            .all(|(i, &e)| i == axis || e == 0)
                    );
                    actual[m.exponents()[axis]] = c.0 / c.1;
                }
                assert_eq!(actual, expected);
            }
        }
    }
}
