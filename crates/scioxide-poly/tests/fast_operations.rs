#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{PolynomialOperationPlan, PolynomialProductAlgorithm};
fn check<C: NumericScalar>() {
    let c = |x: f64| C::from_real(C::Real::try_from_f64_exact(x).unwrap());
    let plan = PolynomialOperationPlan::<C>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 131,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 3,
            inner_len: 3,
        },
        PolynomialOperationRequest::TaylorShift { len: 3 },
    ])
    .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    let a: Vec<_> = (0..129).map(|i| c((i % 7) as f64 - 3.)).collect();
    let b: Vec<_> = (0..131).map(|i| c((i % 5) as f64 - 2.)).collect();
    let mut expected = vec![C::ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            expected[i + j] = expected[i + j] + x * y;
        }
    }
    let mut actual = vec![C::ZERO; expected.len()];
    assert_eq!(
        plan.multiply_into(
            &a,
            &b,
            &mut actual,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap(),
        PolynomialProductAlgorithm::Fft
    );
    let tol = C::Real::try_from_f64_exact(2000.).unwrap() * C::Real::EPSILON;
    for (x, y) in actual.iter().zip(&expected) {
        assert!((*x - *y).abs() < tol);
    }
    let mut result = vec![C::ZERO; 5];
    plan.compose_into(
        &[c(1.), c(2.), c(3.)],
        &[c(1.), c(-1.), c(2.)],
        &mut result,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(result, [c(6.), c(-8.), c(19.), c(-12.), c(12.)]);
    let mut shifted = vec![C::ZERO; 3];
    plan.taylor_shift_into(
        &[c(1.), c(2.), c(3.)],
        c(2.),
        &mut shifted,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(shifted, [c(17.), c(14.), c(3.)]);
    let before = shifted.clone();
    assert!(
        plan.multiply_into(
            &[],
            &[c(1.)],
            &mut shifted,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(before, shifted);
}
#[test]
fn real32() {
    check::<f32>();
}
#[test]
fn real64() {
    check::<f64>();
}
#[test]
fn complex32() {
    check::<Complex<f32>>();
}
#[test]
fn complex64() {
    check::<Complex<f64>>();
}

#[path = "support/exact_field.rs"]
mod exact_field;
#[test]
fn exact_operations() {
    use exact_field::Q;
    use scioxide_poly::{
        ExactPolynomial, ExactPolynomialProductTree, ExactPolynomialWorkspace, exact_multiply_into,
    };
    let q = |x| Q::new(x, 1);
    let p = ExactPolynomial::new(vec![q(1), q(2), q(3)]).unwrap();
    assert_eq!(
        p.taylor_shift(q(2)).unwrap().coefficients(),
        &[q(17), q(14), q(3)]
    );
    let tree = ExactPolynomialProductTree::new(&[q(-1), q(0), q(2)]).unwrap();
    let values = tree.evaluate(&p).unwrap();
    assert_eq!(values, vec![q(2), q(1), q(17)]);
    assert_eq!(tree.interpolate(&values).unwrap(), p);
    let a: Vec<_> = (0..77).map(|i| q(i % 3 - 1)).collect();
    let b: Vec<_> = (0..81).map(|i| q(i % 5 - 2)).collect();
    let mut reference = vec![q(0); a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            reference[i + j] = reference[i + j] + x * y;
        }
    }
    let mut actual = vec![q(0); reference.len()];
    exact_multiply_into(
        &a,
        &b,
        &mut actual,
        &mut ExactPolynomialWorkspace::workspace(reference.len()).unwrap(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(actual, reference);
}
#[test]
#[cfg(feature = "std")]
fn cancellation_and_executor() {
    use scioxide_core::{CancellationToken, ExecutionContext, ExecutionPolicy, ThreadExecutor};
    use scioxide_poly::PolynomialExecutionOptions;
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 256,
        right_len: 256,
    }])
    .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    let a = [1.; 256];
    let mut output = [99.; 511];
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        plan.multiply_into(
            &a,
            &a,
            &mut output,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, [99.; 511]);
    let executor = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let execution = ExecutionContext::new(&executor);
    let parallel = PolynomialOperationPlan::<f64>::build(
        &[PolynomialOperationRequest::Multiply {
            left_len: 256,
            right_len: 256,
        }],
        PolynomialExecutionOptions::new().with_execution_policy(ExecutionPolicy::Auto),
        execution,
    )
    .unwrap();
    let mut parallel_work = parallel.workspace(parallel.requests()).unwrap();
    parallel
        .multiply_into(&a, &a, &mut output, &mut parallel_work, execution)
        .unwrap();
    let mut reference = [0.; 511];
    plan.multiply_into(
        &a,
        &a,
        &mut reference,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (&actual, &expected) in output.iter().zip(&reference) {
        assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * (1.0 + expected.abs()));
    }
}
#[test]
fn dynamic_range_and_complex_cancellation() {
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 129,
        right_len: 129,
    }])
    .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    let mut a = [1.; 129];
    a[0] = 1e200;
    let mut b = [1e-200; 129];
    b[0] = 1.;
    let mut result = [0.; 257];
    assert_eq!(
        plan.multiply_into(
            &a,
            &b,
            &mut result,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap(),
        PolynomialProductAlgorithm::Schoolbook
    );
    assert_eq!(result[0], 1e200);
    assert_eq!(result[1], 2.);
    let before = result;
    a[1] = f64::NAN;
    assert!(
        plan.multiply_into(
            &a,
            &b,
            &mut result,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(result, before);
    let plan =
        PolynomialOperationPlan::<Complex<f64>>::new(&[PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 129,
        }])
        .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    let a = [Complex::new(1., 1.); 129];
    let b = [Complex::new(1., -1.); 129];
    let mut out = [Complex::new(0., 0.); 257];
    plan.multiply_into(
        &a,
        &b,
        &mut out,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    for (i, x) in out.iter().enumerate() {
        let count = (i + 1).min(257 - i).min(129);
        assert!((x.re - 2. * count as f64).abs() < 1e-10);
        assert!(x.im.abs() < 1e-10);
    }
}

fn range_boundaries<R: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let plan = PolynomialOperationPlan::<R>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 9,
            right_len: 9,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 3,
            right_len: 3,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 131,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 1,
            right_len: 1,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 2,
            right_len: 2,
        },
    ])
    .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    // Four-lane native sums overflow before the opposite lanes cancel. Every
    // coefficient of the actual convolution is nevertheless representable.
    let large = R::MAX * R::try_from_f64_exact(0.75).unwrap();
    let a = [large; 9];
    let b = core::array::from_fn::<_, 9, _>(|i| if i % 2 == 0 { R::ONE } else { -R::ONE });
    let mut output = [R::ZERO; 17];
    assert_eq!(
        plan.multiply_into(&a, &b, &mut output, &mut work, context)
            .unwrap(),
        PolynomialProductAlgorithm::Schoolbook
    );
    for (i, &value) in output.iter().enumerate() {
        let expected = if i % 2 == 0 { large } else { R::ZERO };
        assert!((value - expected).abs() <= R::EPSILON * large);
    }

    // Each term is one quarter of a minimum subnormal. Only their sum rounds
    // to a nonzero coefficient, so native product underflow must restart.
    let a = [R::MIN_POSITIVE; 3];
    let b = [R::EPSILON / (R::TWO + R::TWO); 3];
    let mut tiny = [R::ONE; 5];
    plan.multiply_into(&a, &b, &mut tiny, &mut work, context)
        .unwrap();
    let minimum = R::MIN_POSITIVE * R::EPSILON;
    for (i, &value) in tiny.iter().enumerate() {
        let expected = if i == 2 { minimum } else { R::ZERO };
        assert!((value - expected).abs() < minimum);
    }

    // Binary FFT normalization is exact even when one operand is subnormal.
    let (low, high) = if R::MANTISSA_DIGITS == 53 {
        (-1060, 1000)
    } else {
        (-140, 100)
    };
    let a = [R::ONE.scalbn(low); 129];
    let b = [R::ONE.scalbn(high); 131];
    let mut fft = [R::ZERO; 259];
    assert_eq!(
        plan.multiply_into(&a, &b, &mut fft, &mut work, context)
            .unwrap(),
        PolynomialProductAlgorithm::Fft
    );
    for (i, &value) in fft.iter().enumerate() {
        let count = (i + 1).min(259 - i).min(129);
        let expected = R::try_from_usize_exact(count).unwrap().scalbn(low + high);
        assert!(
            (value - expected).abs()
                <= R::EPSILON * R::try_from_f64_exact(4096.).unwrap() * expected
        );
    }

    let before = fft;
    let a = [R::MAX; 129];
    let b = [R::TWO; 131];
    assert!(
        plan.multiply_into(&a, &b, &mut fft, &mut work, context)
            .is_err()
    );
    assert_eq!(fft, before);
    let mut direct = [R::ONE; 1];
    assert!(
        plan.multiply_into(&[R::MAX], &[R::TWO], &mut direct, &mut work, context)
            .is_err()
    );
    assert_eq!(direct, [R::ONE]);
    let mut late_failure = [R::ONE; 3];
    let error = plan
        .multiply_into(
            &[R::ONE, R::MAX],
            &[R::ONE, R::TWO],
            &mut late_failure,
            &mut work,
            context,
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(late_failure, [R::ONE; 3]);
}

#[test]
fn real_product_range_boundaries() {
    range_boundaries::<f32>();
    range_boundaries::<f64>();
}

#[test]
fn complex_products_cancel_overflowing_component_products() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let plan =
        PolynomialOperationPlan::<Complex<f64>>::new(&[PolynomialOperationRequest::Multiply {
            left_len: 1,
            right_len: 1,
        }])
        .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let a = Complex::new(f64::MAX, f64::MAX / 2.);
    let b = Complex::new(1.1, 0.3);
    let mut output = [Complex::new(0., 0.)];
    assert_eq!(
        plan.multiply_into(&[a], &[b], &mut output, &mut workspace, context)
            .unwrap(),
        PolynomialProductAlgorithm::Schoolbook
    );
    let expected = Complex::new(f64::MAX * 0.95, f64::MAX * 0.85);
    assert!((output[0].re - expected.re).abs() <= 4. * f64::EPSILON * f64::MAX);
    assert!((output[0].im - expected.im).abs() <= 4. * f64::EPSILON * f64::MAX);
}

#[test]
fn zero_and_unbalanced_products_keep_direct_accuracy() {
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 257,
            right_len: 3,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 131,
        },
    ])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut a = [0.; 257];
    a[0] = 1e200;
    a[256] = 1e-200;
    let b = [1e-200, 0., 1.];
    let mut product = [0.; 259];
    assert_eq!(
        plan.multiply_into(
            &a,
            &b,
            &mut product,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap(),
        PolynomialProductAlgorithm::Schoolbook
    );
    for (i, &actual) in product.iter().enumerate() {
        let expected: f64 = match i {
            0 => 1.,
            2 => 1e200,
            258 => 1e-200,
            _ => 0.,
        };
        assert!((actual - expected).abs() <= 8. * f64::EPSILON * expected.abs());
    }
    plan.multiply_into(
        &[0.; 129],
        &[1.; 131],
        &mut product,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(product.iter().all(|value| *value == 0.));
}
#[test]
fn resource_failures_and_conditioning() {
    use scioxide_core::ExecutionContext;
    use scioxide_poly::{polynomial_composition_condition, polynomial_product_condition};
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::TaylorShift { len: 3 },
        PolynomialOperationRequest::Multiply {
            left_len: 2,
            right_len: 2,
        },
    ])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut output = [99.; 3];
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert!(
        plan.taylor_shift_into(
            &[1., 2., 3.],
            1.,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, [99.; 3]);
    let mut small = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 1,
        right_len: 1,
    }])
    .unwrap()
    .workspace(&[PolynomialOperationRequest::Multiply {
        left_len: 1,
        right_len: 1,
    }])
    .unwrap();
    assert!(
        plan.multiply_into(
            &[1., 2.],
            &[3., 4.],
            &mut output,
            &mut small,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(output, [99.; 3]);
    assert!(
        PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
            left_len: usize::MAX,
            right_len: 2
        }])
        .is_err()
    );
    let report = polynomial_product_condition(&[1., 1.], &[1., -1.], &[1., 0., -1.]).unwrap();
    assert_eq!(report.condition_estimate(), 2.);
    let report =
        polynomial_composition_condition(&[1., 2., 3.], &[2., 1.], &[17., 14., 3.]).unwrap();
    assert_eq!(report.condition_estimate(), 1.);
}

#[path = "support/prime_field.rs"]
mod prime_field;
#[test]
fn exact_fast_remainders_and_interpolation() {
    use prime_field::Fp;
    use scioxide_poly::{
        ExactPolynomial, ExactPolynomialProductTree, ExactPolynomialWorkspace,
        exact_taylor_shift_into,
    };
    let points: Vec<_> = (0..128).map(Fp::<65537>).collect();
    let tree = ExactPolynomialProductTree::new(&points).unwrap();
    let coefficients: Vec<_> = (0..201).map(|i| Fp::<65537>(i * 13 % 65537)).collect();
    let polynomial = ExactPolynomial::new(coefficients).unwrap();
    let values = tree.evaluate(&polynomial).unwrap();
    for (&x, &y) in points.iter().zip(&values) {
        assert_eq!(y, polynomial.evaluate(x));
    }
    let polynomial =
        ExactPolynomial::new((0..128).map(|i| Fp::<65537>(i * 7 % 65537)).collect()).unwrap();
    let values: Vec<_> = points.iter().map(|&x| polynomial.evaluate(x)).collect();
    assert_eq!(tree.interpolate(&values).unwrap(), polynomial);
    let mut work = ExactPolynomialWorkspace::workspace(128).unwrap();
    let mut shifted = vec![Fp::<65537>(0); 128];
    exact_taylor_shift_into(
        polynomial.coefficients(),
        Fp::<65537>(7),
        &mut shifted,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    let shifted = ExactPolynomial::new(shifted).unwrap();
    for x in [Fp::<65537>(0), Fp::<65537>(11), Fp::<65537>(65530)] {
        assert_eq!(shifted.evaluate(x), polynomial.evaluate(x + Fp::<65537>(7)));
    }
}

#[test]
fn multiply_shapes_are_unordered() {
    use scioxide_core::ExecutionContext;
    let cx = ExecutionContext::sequential();
    // A {2,3} plan serves (3,2) operands with the identical product.
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 2,
        right_len: 3,
    }])
    .unwrap();
    let swapped = [PolynomialOperationRequest::Multiply {
        left_len: 3,
        right_len: 2,
    }];
    assert_eq!(
        plan.workspace_requirements(&swapped).unwrap(),
        plan.workspace_requirements(plan.requests()).unwrap()
    );
    let mut work = plan.workspace(&swapped).unwrap();
    let (a, b) = ([1.0, 2.0], [3.0, -1.0, 0.5]);
    let mut forward = [99.; 4];
    let mut reverse = [99.; 4];
    plan.multiply_into(&a, &b, &mut forward, &mut work, cx)
        .unwrap();
    plan.multiply_into(&b, &a, &mut reverse, &mut work, cx)
        .unwrap();
    assert_eq!(forward, [3.0, 5.0, -1.5, 1.0]);
    assert_eq!(reverse, forward);
    // Large FFT-sized shapes share one preparation in both orders, and a
    // duplicate swapped declaration is merged.
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 100,
            right_len: 101,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 101,
            right_len: 100,
        },
    ])
    .unwrap();
    assert_eq!(plan.requests().len(), 1);
    let mut work = plan.workspace(plan.requests()).unwrap();
    let a: Vec<f64> = (0..100).map(|i| f64::from(i % 7) - 3.0).collect();
    let b: Vec<f64> = (0..101).map(|i| f64::from(i % 5) - 2.0).collect();
    let mut forward = [0.0; 200];
    let mut reverse = [0.0; 200];
    plan.multiply_into(&a, &b, &mut forward, &mut work, cx)
        .unwrap();
    plan.multiply_into(&b, &a, &mut reverse, &mut work, cx)
        .unwrap();
    let scale = 64.0 * f64::EPSILON * 100.0 * 12.0;
    assert!(
        forward
            .iter()
            .zip(&reverse)
            .all(|(x, y)| (x - y).abs() <= scale)
    );
    // An undeclared shape is still rejected in both orders.
    for (left, right) in [(&a[..99], &b[..]), (&b[..99], &a[..])] {
        let error = plan
            .multiply_into(left, right, &mut [0.0; 199], &mut work, cx)
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
}
