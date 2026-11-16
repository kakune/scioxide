//! Division reconstruction, basis matrices and rank-aware polynomial fits.
#![allow(clippy::unwrap_used)]

#[path = "support/tolerance.rs"]
mod tolerance;

use scioxide_core::{
    AlgebraicScalar, Complex, ExecutionContext, NumericScalar, RealScalar, Scaled, ScaledValue,
    SciOxideErrorKind,
};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{
    ApproxPolynomial, BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily,
    PolynomialApproximationPolicy, PolynomialFitPlan, PolynomialOperationPlan, Vandermonde,
};
use tolerance::assert_close;

fn scalar<C: NumericScalar>(x: f64, y: f64) -> C {
    C::from_components(
        C::Real::try_from_f64_exact(x).unwrap(),
        C::Real::try_from_f64_exact(y).unwrap(),
    )
}
fn divisions<C: NumericScalar>() {
    let policy = PolynomialApproximationPolicy::default();
    let b = ApproxPolynomial::monomial_with_options(vec![scalar::<C>(1., 0.25), C::ONE], policy)
        .unwrap();
    let q = ApproxPolynomial::monomial_with_options(
        vec![scalar::<C>(-2., 0.5), scalar::<C>(3., -0.25)],
        policy,
    )
    .unwrap();
    let r = ApproxPolynomial::monomial_with_options(vec![scalar::<C>(0.25, 0.25)], policy).unwrap();
    let a = q.try_mul(&b).unwrap().try_add(&r).unwrap();
    let result = a.div_rem(&b).unwrap();
    for (&actual, &expected) in result
        .quotient()
        .basis()
        .coefficients()
        .iter()
        .zip(q.basis().coefficients())
    {
        assert_close(actual, expected, 128);
    }
    assert_close(
        result.remainder().basis().coefficients()[0],
        r.basis().coefficients()[0],
        128,
    );
    assert!(
        result.condition().backward_error()
            < C::Real::EPSILON * C::Real::try_from_usize_exact(128).unwrap()
    );
    let exact = q.try_mul(&b).unwrap().div_rem(&b).unwrap();
    assert!(
        exact.remainder().basis().coefficients()[0].abs()
            < C::Real::EPSILON * C::Real::try_from_usize_exact(128).unwrap()
    );
    let smaller = r.div_rem(&b).unwrap();
    assert!(smaller.quotient().is_zero());
    assert_eq!(smaller.remainder(), &r);
    let zero = ApproxPolynomial::monomial_with_options(vec![C::ZERO], policy).unwrap();
    assert!(zero.div_rem(&b).unwrap().quotient().is_zero());
    assert_eq!(
        a.div_rem(&zero).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let plan = PolynomialOperationPlan::<C>::new(&[PolynomialOperationRequest::Divide {
        dividend_len: 3,
        divisor_len: 2,
    }])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut quotient = [C::ZERO; 2];
    let mut remainder = [C::ZERO; 1];
    for _ in 0..2 {
        {
            let report = plan
                .div_rem_into(
                    a.basis().coefficients(),
                    b.basis().coefficients(),
                    &mut quotient,
                    &mut remainder,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.backward_error() >= C::Real::ZERO);
        };
    }
}
#[test]
fn division_all_scalar_domains() {
    divisions::<f32>();
    divisions::<f64>();
    divisions::<Complex<f32>>();
    divisions::<Complex<f64>>();
}

#[test]
fn division_retains_extended_intermediates_and_tiny_coefficients() {
    let policy = PolynomialApproximationPolicy::default();
    let h = f64::MAX * 0.75;
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Divide {
        dividend_len: 3,
        divisor_len: 2,
    }])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut q = [0.; 2];
    let mut r = [0.; 1];
    {
        let report = plan
            .div_rem_into(
                &[-h, -h, h],
                &[h, h],
                &mut q,
                &mut r,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_close(q[0], -2., 8);
        assert_close(q[1], 1., 8);
        assert_close(r[0] / h, 1., 8);
        assert!(report.backward_error() < 1e-14);
    };
    let tiny = 2.0_f64.powi(-1000);
    let large = 2.0_f64.powi(1000);
    let a = ApproxPolynomial::monomial_with_options(vec![tiny, -large, large], policy).unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![1., -1.], policy).unwrap();
    let result = a.div_rem(&b).unwrap();
    assert_close(result.remainder().basis().coefficients()[0] / tiny, 1., 8);
    assert_close(result.quotient().basis().coefficients()[1] / large, -1., 8);
}

#[test]
fn division_report_matches_independent_scaled_reconstruction() {
    let policy = PolynomialApproximationPolicy::default();
    let a = ApproxPolynomial::monomial_with_options(vec![0.3_f64, -0.7, 1.1, 0.9, 0.2], policy)
        .unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![0.1, 0.7, 0.3], policy).unwrap();
    let result = a.div_rem(&b).unwrap();
    let q = result.quotient().basis().coefficients();
    let r = result.remainder().basis().coefficients();
    let mut residual = Scaled::ZERO;
    for (k, &value) in a.basis().coefficients().iter().enumerate() {
        let mut reconstructed =
            ScaledValue::<f64>::from_value(r.get(k).copied().unwrap_or(0.)).unwrap();
        for (i, &x) in q.iter().enumerate() {
            if let Some(&y) = k
                .checked_sub(i)
                .and_then(|j| b.basis().coefficients().get(j))
            {
                reconstructed = reconstructed
                    .add(
                        ScaledValue::from_value(x)
                            .unwrap()
                            .mul(ScaledValue::from_value(y).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
        residual = residual
            .add(
                ScaledValue::from_value(value)
                    .unwrap()
                    .sub(reconstructed)
                    .unwrap()
                    .magnitude()
                    .unwrap(),
            )
            .unwrap();
    }
    let norm = |x: &[f64]| x.iter().map(|v| v.abs()).sum::<f64>();
    let error = residual.round().unwrap()
        / (norm(a.basis().coefficients()) + norm(q) * norm(b.basis().coefficients()) + norm(r));
    assert!((error - result.condition().backward_error()).abs() < 8. * f64::EPSILON);
    assert!(result.condition().condition_estimate() >= 1.);
}

#[test]
fn division_and_matrix_errors_preserve_outputs() {
    let plan = PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::Divide {
        dividend_len: 3,
        divisor_len: 2,
    }])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut q = [99.; 2];
    let mut r = [99.];
    for b in [&[0., 0.][..], &[f64::NAN, 1.][..], &[1., 0.][..]] {
        assert!(
            plan.div_rem_into(
                &[1., 2., 3.],
                b,
                &mut q,
                &mut r,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(q, [99.; 2]);
        assert_eq!(r, [99.]);
    }
    assert_eq!(
        plan.div_rem_into(
            &[1., 2., 3.],
            &[1., 1.],
            &mut q,
            &mut r,
            &mut Default::default(),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let basis = Vandermonde::monomial(2).unwrap();
    let mut out = scioxide_linalg::Matrix::from_vec(2, 3, vec![99.; 6]).unwrap();
    assert!(
        basis
            .matrix_into(
                &[0., f64::NAN],
                out.view_mut(),
                &mut [0.; 6],
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(out.as_slice(), &[99.; 6]);
    assert_eq!(
        basis
            .matrix_into(
                &[0., 1.],
                out.view_mut(),
                &mut [0.; 5],
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn every_orthogonal_matrix_matches_series_evaluation() {
    use OrthogonalFamily as F;
    for family in [
        F::ChebyshevT,
        F::ChebyshevU,
        F::Legendre,
        F::HermiteH,
        F::HermiteHe,
        F::Laguerre,
        F::GeneralizedLaguerre { alpha: 0.5 },
        F::Jacobi {
            alpha: 0.25,
            beta: 0.5,
        },
        F::Gegenbauer { lambda: 0.75 },
    ] {
        for normalization in [
            BasisNormalization::Classical,
            BasisNormalization::Monic,
            BasisNormalization::Orthonormal,
        ] {
            let spec = OrthogonalBasisSpec::new(family, normalization, 2.0_f64, 3.).unwrap();
            let prepared = spec.prepare(5, Default::default()).unwrap();
            let plan = Vandermonde::orthogonal(prepared.clone());
            let points = [
                Complex::new(1., 0.25),
                Complex::new(2., -0.25),
                Complex::new(3., 0.),
            ];
            let matrix = plan.matrix(&points).unwrap();
            for (row, &point) in points.iter().enumerate() {
                for k in 0..6 {
                    let mut coefficients = [Complex::new(0., 0.); 6];
                    coefficients[k] = Complex::new(1., 0.);
                    assert_close(
                        matrix.as_slice()[row * 6 + k],
                        prepared
                            .evaluate_with_derivative(&coefficients, point)
                            .unwrap()
                            .value(),
                        2048,
                    );
                }
            }
        }
    }
    let empty = Vandermonde::<f64>::monomial(0)
        .unwrap()
        .matrix(&[] as &[f64])
        .unwrap();
    assert_eq!((empty.rows(), empty.cols()), (0, 1));
}

fn fitting<C: NumericScalar>() {
    let points = [-1., -0.5, 0., 0.5, 1.].map(|x| scalar::<C>(x, 0.));
    let expected = [
        scalar::<C>(1., 0.25),
        scalar::<C>(2., -0.5),
        scalar::<C>(-0.5, 0.25),
    ];
    let values = points.map(|x| expected[0] + x * (expected[1] + x * expected[2]));
    let plan = PolynomialFitPlan::monomial(&points, 2, None, Default::default()).unwrap();
    let result = plan.fit(&values).unwrap();
    for (&actual, &wanted) in result
        .polynomial()
        .basis()
        .coefficients()
        .iter()
        .zip(&expected)
    {
        assert_close(actual, wanted, 256);
    }
    assert_eq!(result.report().rank().rank(), 3);
    let mut workspace = plan.workspace(()).unwrap();
    let mut out = [C::ZERO; 3];
    for _ in 0..2 {
        {
            let _ = plan
                .fit_into(
                    &values,
                    &mut out,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
        };
    }
    assert!(
        plan.fit_into(
            &values,
            &mut out[..2],
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    let snapshot = out;
    let mut bad = values;
    bad[2] = scalar::<C>(0., 0.) / C::ZERO;
    assert!(
        plan.fit_into(
            &bad,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, snapshot);
}
#[test]
fn fitting_all_scalar_domains() {
    fitting::<f32>();
    fitting::<f64>();
    fitting::<Complex<f32>>();
    fitting::<Complex<f64>>();
}

#[test]
fn weighted_fit_matches_two_by_two_normal_equations() {
    let x = [-1.0_f64, 0., 1., 2.];
    let y = [0.1, 1.2, 2.4, 2.8];
    let w = [1., 0., 2., 4.];
    let plan = PolynomialFitPlan::monomial(&x, 1, Some(&w), Default::default()).unwrap();
    let fit = plan.fit(&y).unwrap();
    let s0: f64 = w.iter().sum();
    let s1: f64 = x.iter().zip(w).map(|(x, w)| x * w).sum();
    let s2: f64 = x.iter().zip(w).map(|(x, w)| x * x * w).sum();
    let t0: f64 = y.iter().zip(w).map(|(y, w)| y * w).sum();
    let t1: f64 = x.iter().zip(y).zip(w).map(|((x, y), w)| x * y * w).sum();
    let determinant = s0 * s2 - s1 * s1;
    let wanted = [
        (t0 * s2 - t1 * s1) / determinant,
        (t1 * s0 - t0 * s1) / determinant,
    ];
    for (&actual, wanted) in fit.polynomial().basis().coefficients().iter().zip(wanted) {
        assert_close(actual, wanted, 256);
    }
    let residual = x
        .iter()
        .zip(y)
        .zip(w)
        .map(|((&x, y), w)| w * (wanted[0] + wanted[1] * x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_close(fit.report().residual_norm(), residual, 256);
    assert!(PolynomialFitPlan::monomial(&x, 1, Some(&[0.; 4]), Default::default()).is_err());
    assert!(PolynomialFitPlan::monomial(&x, 1, Some(&[-1.; 4]), Default::default()).is_err());
}

#[test]
fn rank_deficiency_and_basis_conditioning_are_reported() {
    let plan = PolynomialFitPlan::monomial(&[1.0_f64; 5], 2, None, Default::default()).unwrap();
    let fit = plan.fit(&[3.; 5]).unwrap();
    assert_eq!(fit.report().rank().rank(), 1);
    assert_close(fit.polynomial().evaluate(1.).unwrap(), 3., 128);
    let points: Vec<_> = (0..41).map(|k| -1. + k as f64 / 20.).collect();
    let mono = PolynomialFitPlan::monomial(&points, 14, None, Default::default())
        .unwrap()
        .fit(&vec![1.; points.len()])
        .unwrap();
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.,
        1.,
    )
    .unwrap();
    let cheb = PolynomialFitPlan::orthogonal(
        &points,
        spec.prepare(14, Default::default()).unwrap(),
        None,
        Default::default(),
    )
    .unwrap()
    .fit(&vec![1.; points.len()])
    .unwrap();
    assert!(
        cheb.report().condition().reciprocal().value()
            > 1000. * mono.report().condition().reciprocal().value()
    );
    assert_eq!(cheb.polynomial().basis().plan().spec(), spec);
    let wide = PolynomialFitPlan::monomial(&[0.0_f64], 2, None, Default::default())
        .unwrap()
        .fit(&[2.])
        .unwrap();
    assert_close(wide.polynomial().basis().coefficients()[0], 2., 64);
}

#[test]
fn strided_matrices_and_prepared_fallbacks_need_no_allocation() {
    let plan = Vandermonde::<f64>::monomial(2).unwrap();
    let mut storage = [99.; 12];
    let mut scratch = [0.; 6];
    {
        let view =
            scioxide_linalg::MatrixViewMut::from_slice_with_strides(2, 3, &mut storage, 6, 2)
                .unwrap();
        plan.matrix_into(
            &[0., 2.],
            view,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    };
    for (&actual, expected) in storage.iter().step_by(2).zip([1., 0., 0., 1., 2., 4.]) {
        assert_close(actual, expected, 8);
    }
    assert!(storage.iter().skip(1).step_by(2).all(|&x| x == 99.));
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::HermiteH,
        BasisNormalization::Classical,
        0.,
        1.,
    )
    .unwrap();
    let plan = Vandermonde::orthogonal(spec.prepare(200, Default::default()).unwrap());
    let mut matrix = scioxide_linalg::Matrix::zeros(1, 201).unwrap();
    let mut scratch = vec![0.; 201];
    {
        plan.matrix_into(
            &[0.0_f64],
            matrix.view_mut(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    };
    assert_close(
        matrix.as_slice()[200] / OrthogonalFamily::HermiteH.evaluate(200, 0.0_f64).unwrap(),
        1.,
        4096,
    );
}

#[test]
fn division_sensitivity_detects_unstable_exact_cancellation() {
    let mut coefficients = vec![0.; 14];
    coefficients[12] = -2.;
    coefficients[13] = 1.;
    let a = ApproxPolynomial::monomial(coefficients).unwrap();
    let b = ApproxPolynomial::monomial(vec![-2., 1.]).unwrap();
    let result = a.div_rem(&b).unwrap();
    assert!(result.remainder().is_zero());
    assert!(result.condition().condition_estimate() > 1000.);
    assert!(result.condition().backward_error() < 1e-14);
}

#[test]
fn extreme_weights_and_rhs_scales_preserve_finite_fit_coefficients() {
    for scale in [2.0_f64.powi(-900), 2.0_f64.powi(900)] {
        let plan = PolynomialFitPlan::monomial(
            &[-1.0_f64, 0., 1.],
            1,
            Some(&[scale; 3]),
            Default::default(),
        )
        .unwrap();
        let result = plan.fit(&[-1., 1., 3.]).unwrap();
        assert_close(result.polynomial().basis().coefficients()[0], 1., 256);
        assert_close(result.polynomial().basis().coefficients()[1], 2., 256);
    }
    let plan =
        PolynomialFitPlan::monomial(&[-1.0_f64, 0., 1.], 1, None, Default::default()).unwrap();
    let result = plan.fit(&[-1e300, 1e300, 3e300]).unwrap();
    assert_close(
        result.polynomial().basis().coefficients()[0] / 1e300,
        1.,
        256,
    );
    assert_close(
        result.polynomial().basis().coefficients()[1] / 1e300,
        2.,
        256,
    );
}

fn fitting_range_loss<C: NumericScalar>() {
    let real = C::from_real;
    let number = |x| C::Real::try_from_f64_lossy(x).unwrap();
    let minimum = C::Real::MIN_POSITIVE;
    let maximum = C::Real::MAX;
    for (point, weight) in [(minimum, minimum), (maximum.sqrt(), maximum)] {
        let plan = PolynomialFitPlan::monomial(
            &[-real(point), real(point)],
            1,
            Some(&[weight; 2]),
            Default::default(),
        )
        .unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let mut coefficients = [C::ZERO; 2];
        for amplitude in [C::Real::ONE, C::Real::ONE / number(2.)] {
            let values = [-real(amplitude), real(amplitude)];
            {
                let report = plan
                    .fit_into(
                        &values,
                        &mut coefficients,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert_eq!(report.rank().rank(), 2);
                assert_close(coefficients[0], C::ZERO, 64);
                assert_close(coefficients[1] * real(point), real(amplitude), 256);
            };
        }
    }

    // Weighting these observations underflows before RHS normalization can
    // recover them. Final coefficients are still ordinary finite values.
    let plan = PolynomialFitPlan::monomial(
        &[-C::ONE, C::ONE],
        1,
        Some(&[minimum; 2]),
        Default::default(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = [C::ZERO; 2];
    {
        let report = plan
            .fit_into(
                &[-real(minimum), real(minimum)],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.rank().rank(), 2);
        assert_close(output[1] / real(minimum), C::ONE, 256);
    };

    // The constant fit is 0.45*MAX. Its difference from the second observation
    // is 1.2*MAX in each component, but the weighted residual is representable.
    let positive = C::from_components(maximum * number(0.75), maximum * number(0.75));
    let weights = [minimum * number(4.), minimum];
    let plan =
        PolynomialFitPlan::monomial(&[C::ZERO; 2], 0, Some(&weights), Default::default()).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = [C::ZERO];
    let expected = maximum * minimum.sqrt() * number(if C::IS_REAL { 1.8 } else { 3.6 }).sqrt();
    {
        let report = plan
            .fit_into(
                &[positive, -positive],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_close(
            output[0] / real(maximum),
            C::from_components(number(0.45), number(0.45)),
            256,
        );
        assert_close(report.residual_norm() / expected, C::Real::ONE, 256);
    };

    // A zero-weight row is omitted even if its unweighted prediction overflows.
    let plan = PolynomialFitPlan::monomial(
        &[real(C::Real::ONE), real(maximum)],
        1,
        Some(&[C::Real::ONE, C::Real::ZERO]),
        Default::default(),
    )
    .unwrap();
    let fit = plan.fit(&[real(number(8.)), C::ZERO]).unwrap();
    assert_close(fit.report().residual_norm(), C::Real::ZERO, 128);

    // An unrepresentable final norm is an error and cannot publish coefficients.
    let policy = PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::from_sqrt_epsilon(),
        scioxide_linalg::RankPolicy::Threshold(
            scioxide_core::RankThreshold::new(number(1.1), C::Real::ZERO).unwrap(),
        ),
    );
    let plan = PolynomialFitPlan::monomial(&[C::ZERO; 2], 0, None, policy).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = [C::ONE];
    let error = plan
        .fit_into(
            &[real(maximum); 2],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    assert_eq!(output, [C::ONE]);
}

#[test]
fn fitting_range_loss_preserves_results_and_reserved_allocation_contract() {
    fitting_range_loss::<f32>();
    fitting_range_loss::<f64>();
    fitting_range_loss::<Complex<f32>>();
    fitting_range_loss::<Complex<f64>>();
}

#[test]
fn explicit_rank_cutoff_and_complex_orthogonal_fit_preserve_the_policy() {
    let policy = PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::from_sqrt_epsilon(),
        scioxide_linalg::RankPolicy::Threshold(scioxide_core::RankThreshold::new(1.1, 0.).unwrap()),
    );
    let fit = PolynomialFitPlan::monomial(&[-1.0_f64, 0., 1.], 1, None, policy)
        .unwrap()
        .fit(&[-1., 1., 3.])
        .unwrap();
    assert_eq!(fit.report().rank().rank(), 0);
    assert!(fit.polynomial().is_zero());
    assert_eq!(fit.report().policy(), policy);
    assert_close(fit.report().residual_norm(), 11.0_f64.sqrt(), 128);

    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Orthonormal,
        2.0_f64,
        3.,
    )
    .unwrap();
    let points = [-1., 0., 1., 2., 3., 4., 5.].map(|x| Complex::new(x, 0.125));
    let prepared = spec.prepare(2, Default::default()).unwrap();
    let expected = [
        Complex::new(1., 0.5),
        Complex::new(-0.5, 0.25),
        Complex::new(0.25, -0.125),
    ];
    let values = points.map(|x| {
        prepared
            .evaluate_with_derivative(&expected, x)
            .unwrap()
            .value()
    });
    let fit = PolynomialFitPlan::orthogonal(&points, prepared, None, Default::default())
        .unwrap()
        .fit(&values)
        .unwrap();
    for (&actual, wanted) in fit.polynomial().basis().coefficients().iter().zip(expected) {
        assert_close(actual, wanted, 512);
    }
    assert_eq!(fit.polynomial().basis().plan().spec(), spec);
}
