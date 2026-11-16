//! First-use and reused scratch coverage for all prepared kernels.
#![allow(clippy::unwrap_used)]

use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar};
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{
    ExactPolynomialWorkspace, PolynomialOperationPlan, exact_multiply_into, exact_taylor_shift_into,
};
#[path = "support/exact_field.rs"]
mod exact_field;

#[test]
fn swapped_multiply_operands_reuse_the_prepared_shape_without_allocation() {
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Multiply {
        left_len: 128,
        right_len: 129,
    }])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let (a, b) = ([1.0; 128], [0.5; 129]);
    let mut forward = [0.0; 256];
    let mut reverse = [0.0; 256];
    plan.multiply_into(
        &a,
        &b,
        &mut forward,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    {
        plan.multiply_into(
            &b,
            &a,
            &mut reverse,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    };
    assert!(
        forward
            .iter()
            .zip(&reverse)
            .all(|(x, y)| (x - y).abs() <= 1e-10 * x.abs().max(1.0))
    );
}

#[test]
fn exact_shape_queries_and_reservations_do_not_allocate() {
    use PolynomialOperationRequest::*;
    let requests = [
        Multiply {
            left_len: 65,
            right_len: 65,
        },
        Divide {
            dividend_len: 129,
            divisor_len: 2,
        },
    ];
    let plan = PolynomialOperationPlan::<f64>::new(&requests).unwrap();
    let requirements = plan.workspace_requirements(&requests).unwrap();
    let mut workspace = plan.workspace(&requests).unwrap();
    {
        assert_eq!(
            plan.workspace_requirements(&requests).unwrap(),
            requirements
        );
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            scioxide_core::Capacity::Sufficient
        );
        workspace.try_reserve(&requirements).unwrap();
    };
    let mut output = [99.; 127];
    let error = plan
        .multiply_into(
            &[1.; 64],
            &[1.; 64],
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    assert_eq!(output, [99.; 127]);
    let mut quotient = [99.; 128];
    let mut remainder = [99.];
    let mut product_only = plan.workspace(&requests[..1]).unwrap();
    let error = plan
        .div_rem_into(
            &[1.; 129],
            &[1.; 2],
            &mut quotient,
            &mut remainder,
            &mut product_only,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(quotient, [99.; 128]);
    assert_eq!(remainder, [99.]);
}

fn composition_storage<C: NumericScalar>() {
    let real = |value| C::Real::try_from_f64_exact(value).unwrap();
    for outer_len in [1, 2, 3, 5, 16, 31, 32, 33, 64, 65, 129, 512, 513, 1024] {
        for inner_len in [1, 2, 3, 4] {
            let request = PolynomialOperationRequest::Compose {
                outer_len,
                inner_len,
            };
            let plan = PolynomialOperationPlan::<C>::new(&[request]).unwrap();
            let mut workspace = plan.workspace(&[request]).unwrap();
            let larger =
                PolynomialOperationPlan::<C>::new(&[PolynomialOperationRequest::Compose {
                    outer_len: outer_len + 1,
                    inner_len,
                }])
                .unwrap();
            let larger_requirements = larger.workspace_requirements(larger.requests()).unwrap();
            let p = vec![C::ONE; outer_len];
            let mut q = vec![C::ZERO; inner_len];
            q[0] = C::from_real(real(0.125));
            if inner_len > 1 {
                q[inner_len - 1] = C::from_components(real(0.25), real(0.125));
            }
            let mut output = vec![C::ZERO; (outer_len - 1) * (inner_len - 1) + 1];
            for use_index in 0..2 {
                if use_index == 1 {
                    // Exercise exactly counted first-use storage, then a larger
                    // retained workspace, with no allocation during either call.
                    workspace.try_reserve(&larger_requirements).unwrap();
                }
                {
                    plan.compose_into(
                        &p,
                        &q,
                        &mut output,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                };
            }
            for x in [C::from_real(real(-0.25)), C::from_real(real(0.5))] {
                let evaluate = |coefficients: &[C], point| {
                    coefficients
                        .iter()
                        .rev()
                        .fold(C::ZERO, |sum, &c| sum * point + c)
                };
                let expected = evaluate(&p, evaluate(&q, x));
                let actual = evaluate(&output, x);
                assert!(
                    (actual - expected).abs()
                        <= real(4096.) * C::Real::EPSILON * (C::Real::ONE + expected.abs())
                );
            }
        }
    }
}

#[test]
fn composition_fits_reserved_peak_storage_in_all_scalar_domains() {
    composition_storage::<f32>();
    composition_storage::<f64>();
    composition_storage::<Complex<f32>>();
    composition_storage::<Complex<f64>>();
}

fn product_allocations<C: NumericScalar>() {
    let plan = PolynomialOperationPlan::<C>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 129,
        },
        PolynomialOperationRequest::Multiply {
            left_len: 9,
            right_len: 9,
        },
    ])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let one = C::from_components(C::Real::ONE, C::Real::ONE);
    let a = [one; 129];
    let mut fft = [C::ZERO; 257];
    let large = C::from_real(C::Real::MAX * C::Real::HALF);
    let direct = [large; 9];
    let signs = core::array::from_fn::<_, 9, _>(|i| if i % 2 == 0 { C::ONE } else { -C::ONE });
    let mut fallback = [C::ZERO; 17];
    for _ in 0..2 {
        {
            plan.multiply_into(
                &a,
                &a,
                &mut fft,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        };
        {
            plan.multiply_into(
                &direct,
                &signs,
                &mut fallback,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        };
    }
}

#[test]
fn first_and_reused_real_and_complex_products_do_not_allocate() {
    product_allocations::<f32>();
    product_allocations::<f64>();
    product_allocations::<Complex<f32>>();
    product_allocations::<Complex<f64>>();
}
#[test]
fn monomial_evaluation_executes_in_either_range() {
    for scale in [1.0_f64, 2.0_f64.powi(1000), 2.0_f64.powi(-1000)] {
        let coefficients = [scale, -scale, scale];
        let polynomial = scioxide_poly::ApproxPolynomial::monomial(coefficients.to_vec()).unwrap();
        {
            let _ = std::hint::black_box(polynomial.evaluate_with_derivative(1.0).unwrap());
        };
        {
            let checked = scioxide_poly::checked_basis_evaluation(
                &coefficients,
                scioxide_core::Complex::new(1.0, 0.0),
                scioxide_poly::PolynomialRootBasisRef::Monomial,
            )
            .unwrap();
            std::hint::black_box(checked.root_condition().unwrap());
            std::hint::black_box(checked);
        };
    }
}
#[test]
fn prepared_operations_do_not_allocate() {
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 129,
            right_len: 129,
        },
        PolynomialOperationRequest::TaylorShift { len: 129 },
    ])
    .unwrap();
    let mut work = plan.workspace(plan.requests()).unwrap();
    let a = [1.; 129];
    let mut product = [0.; 257];
    let mut shifted = [0.; 129];
    for _ in 0..2 {
        {
            plan.multiply_into(
                &a,
                &a,
                &mut product,
                &mut work,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        };
        {
            plan.taylor_shift_into(
                &a,
                0.01,
                &mut shifted,
                &mut work,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        };
    }
    let mut exact_work = ExactPolynomialWorkspace::workspace(129).unwrap();
    let a = [exact_field::Q(1, 1); 65];
    let mut out = [exact_field::Q(0, 1); 129];
    {
        exact_taylor_shift_into(
            &a,
            exact_field::Q(1, 1),
            &mut out[..65],
            &mut exact_work,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
    {
        exact_multiply_into(
            &a,
            &a,
            &mut out,
            &mut exact_work,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
    for _ in 0..2 {
        {
            exact_multiply_into(
                &a[..64],
                &a[..32],
                &mut out[..95],
                &mut exact_work,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        };
    }
}
