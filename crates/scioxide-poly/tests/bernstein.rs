//! Direct rational Bernstein formula is independent of de Casteljau evaluation.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_poly::*;
#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;
fn binomial(n: usize, k: usize) -> i128 {
    (0..k).fold(1, |v, i| v * (n - i) as i128 / (i + 1) as i128)
}
fn pow(x: Q, n: usize) -> Q {
    (0..n).fold(Q(1, 1), |v, _| v * x)
}
fn basis(n: usize, i: usize, t: Q) -> Q {
    Q(binomial(n, i), 1) * pow(t, i) * pow(Q(1, 1) - t, n - i)
}
#[test]
fn rational_tensor_oracle_both_widths() {
    fn check<R: RealScalar>() {
        let r = |x| R::try_from_f64_lossy(x).unwrap();
        let coefficients: Vec<_> = (0..18).map(|i| r(f64::from(i % 7) - 3.)).collect();
        let net = TensorBernstein::new(
            &[2, 2],
            vec![
                Interval::new(r(-2.), r(2.)).unwrap(),
                Interval::new(r(0.), r(2.)).unwrap(),
            ],
            2,
            coefficients,
        )
        .unwrap();
        let mut scratch = vec![
            R::ZERO;
            net.as_view()
                .workspace_requirements(1)
                .unwrap()
                .scratch_len()
        ];
        let mut output = [R::ZERO; 2];
        for u in 0..=8 {
            for v in 0..=8 {
                let uq = Q(u, 8);
                let vq = Q(v, 8);
                net.as_view()
                    .evaluate_into(
                        &[r(u as f64 / 2. - 2.), r(v as f64 / 4.)],
                        &mut output,
                        &mut scratch,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                for (component, &actual) in output.iter().enumerate() {
                    let mut expected = Q(0, 1);
                    for i in 0..3 {
                        for j in 0..3 {
                            let c = ((i * 3 + j) * 2 + component) % 7;
                            expected =
                                expected + Q(c as i128 - 3, 1) * basis(2, i, uq) * basis(2, j, vq);
                        }
                    }
                    let expected = r(expected.0 as f64 / expected.1 as f64);
                    assert!((actual - expected).abs() <= r(32.) * R::EPSILON);
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn subdivision_elevation_and_derivatives() {
    fn check<C: NumericScalar>() {
        let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
        let c = |x| C::from_real(r(x));
        let net = TensorBernstein::new(
            &[2, 1],
            vec![
                Interval::new(r(-1.), r(3.)).unwrap(),
                Interval::new(r(2.), r(4.)).unwrap(),
            ],
            1,
            vec![c(0.), c(2.), c(1.), c(3.), c(4.), c(6.)],
        )
        .unwrap();
        let (left, right) = net.subdivide(0, r(1.)).unwrap();
        let elevated = net.elevate_degree(1, 5).unwrap();
        let derivative = net.derivative(1).unwrap();
        let eval = |p: &TensorBernstein<C>, x: f64, y: f64| {
            let mut out = [C::ZERO];
            let view = p.as_view();
            view.evaluate_into(
                &[c(x), c(y)],
                &mut out,
                &mut vec![C::ZERO; view.workspace_requirements(1).unwrap().scratch_len()],
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            out[0]
        };
        for x in [-1., 0., 1., 2., 3.] {
            for y in [2., 2.5, 3., 4.] {
                let expected = c((x + 1.) * (x + 1.) / 8. + (x + 1.) / 2. + y - 2.);
                let actual = eval(&net, x, y);
                let tol = r(128.) * C::Real::EPSILON;
                assert!((actual - expected).abs() <= tol);
                assert!((eval(if x <= 1. { &left } else { &right }, x, y) - actual).abs() <= tol);
                assert!((eval(&elevated, x, y) - actual).abs() <= tol);
                assert!((eval(&derivative, x, y) - c(1.)).abs() <= tol);
            }
        }
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
#[test]
fn univariate_geometry_and_range_safety() {
    let p = ApproxPolynomial::bernstein(vec![0., 2., 0.], -1., 1.).unwrap();
    let (left, right) = p.subdivide(0.).unwrap();
    let raised = p.elevate_degree(7).unwrap();
    for i in 0..=16 {
        let x = f64::from(i) / 8. - 1.;
        let expected = 1. - x * x;
        assert!((raised.evaluate(x).unwrap() - expected).abs() < 1e-14);
        assert!(
            ((if x < 0. { &left } else { &right }).evaluate(x).unwrap() - expected).abs() < 1e-14
        );
    }
    assert!(p.subdivide(-1.).is_err());
    assert!(p.elevate_degree(1).is_err());
    assert!(p.elevate_degree(usize::MAX).is_err());
    let b = p.control_bounds().unwrap();
    assert_eq!(b.coordinates()[0], Interval::new(0., 2.).unwrap());
    // The former direct blend overflows outside [0,1] even when cancellation yields a finite constant.
    let huge = ApproxPolynomial::bernstein(vec![f64::MAX, f64::MAX], 0., 1.).unwrap();
    let tensor = TensorBernstein::new(
        &[1],
        vec![Interval::new(0., 1.).unwrap()],
        1,
        vec![f64::MAX, f64::MAX],
    )
    .unwrap();
    let mut output = [0.];
    tensor
        .as_view()
        .evaluate_into(
            &[2.],
            &mut output,
            &mut [0.; 3],
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [f64::MAX]);
    assert_eq!(
        huge.subdivide(0.5).unwrap().0.basis().coefficients(),
        [f64::MAX, f64::MAX]
    );
}
#[test]
fn geometric_boxes_and_sparse_equivalence() {
    // Bilinear surface (u,v,u*v) has four corners and lies in their coordinate box.
    let net = TensorBernstein::new(
        &[1, 1],
        vec![Interval::new(0., 1.).unwrap(); 2],
        3,
        vec![0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1.],
    )
    .unwrap();
    let sparse = ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::Lexicographic,
        vec![(Monomial::new(&[1, 1]).unwrap(), 1.)],
    )
    .unwrap();
    let bounds = net.control_bounds().unwrap();
    let (left, right) = net.subdivide(1, 0.25).unwrap();
    for child in [&left, &right] {
        for (small, big) in child
            .control_bounds()
            .unwrap()
            .coordinates()
            .iter()
            .zip(bounds.coordinates())
        {
            assert!(small.lower() >= big.lower());
            assert!(small.upper() <= big.upper());
        }
    }
    for i in 0..=8 {
        for j in 0..=8 {
            let point = [f64::from(i) / 8., f64::from(j) / 8.];
            let mut output = [0.; 3];
            net.as_view()
                .evaluate_into(
                    &point,
                    &mut output,
                    &mut [0.; 7],
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(output[2], sparse.evaluate(&point).unwrap());
            for (&c, b) in output.iter().zip(bounds.coordinates()) {
                assert!(b.contains(c));
            }
        }
    }
}
#[test]
fn borrowed_shapes_constants_and_failure_transaction() {
    let domains = [Interval::new(0., 1.).unwrap()];
    let c = [1., 2.];
    let view = TensorBernsteinView::new(
        &[2],
        &domains,
        1,
        &c,
        scioxide_poly::PolynomialExecutionOptions::new(),
    )
    .unwrap();
    let mut out = [17.];
    let mut work = [0.; 3];
    assert_eq!(view.workspace_requirements(1).unwrap().scratch_len(), 3);
    let short = view
        .evaluate_into(
            &[0.5],
            &mut out,
            &mut work[..2],
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        short.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(out, [17.]);
    view.evaluate_into(
        &[0.5],
        &mut out,
        &mut work,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out, [1.5]);
    assert!(
        view.evaluate_into(
            &[f64::NAN],
            &mut out,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [1.5]);
    assert!(
        TensorBernsteinView::new(
            &[0],
            &domains,
            1,
            &[] as &[f64],
            scioxide_poly::PolynomialExecutionOptions::new()
        )
        .is_err()
    );
    assert!(TensorBernstein::<f64>::new(&[usize::MAX], domains.to_vec(), 1, vec![]).is_err());
    let constant = TensorBernstein::new(&[], vec![], 2, vec![3., 4.]).unwrap();
    let mut out = [0.; 2];
    constant
        .as_view()
        .evaluate_into(
            &[],
            &mut out,
            &mut [0.; 3],
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out, [3., 4.]);
    let complex = TensorBernstein::new(
        &[1],
        domains.to_vec(),
        1,
        vec![Complex::new(1., 1.), Complex::new(2., -1.)],
    )
    .unwrap();
    let mut out = [Complex::new(0., 0.)];
    complex
        .as_view()
        .evaluate_into(
            &[Complex::new(0., 1.)],
            &mut out,
            &mut [Complex::new(0., 0.); 3],
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out, [Complex::new(3., 2.)]);
}

#[test]
fn extrapolation_complex_points_and_derivatives_match_quadratic_formula() {
    fn check<C: NumericScalar>() {
        let real = |x| C::Real::try_from_f64_lossy(x).unwrap();
        let c = |re, im| C::from_components(real(re), real(im));
        let controls = [c(1., 0.5), c(-2., -0.25), c(4., 1.)];
        let p = ApproxPolynomial::bernstein(controls.to_vec(), real(-2.), real(2.)).unwrap();
        for re in [-2., -0.25, 0., 0.375, 1., 1.25, 2.] {
            for im in [0., 0.25] {
                let t = c(re, im);
                let u = C::ONE - t;
                let expected =
                    controls[0] * u * u + c(2., 0.) * controls[1] * t * u + controls[2] * t * t;
                let derivative =
                    ((controls[1] - controls[0]) * u + (controls[2] - controls[1]) * t) / c(2., 0.);
                let result = p
                    .evaluate_with_derivative(c(-2., 0.) + c(4., 0.) * t)
                    .unwrap();
                let tolerance = real(128.) * C::Real::EPSILON;
                assert!(
                    (result.value() - expected).abs()
                        <= tolerance * expected.abs().max(C::Real::ONE)
                );
                assert!(
                    (result.derivative() - derivative).abs()
                        <= tolerance * derivative.abs().max(C::Real::ONE)
                );
            }
        }
        let mut short = [C::ZERO; 3];
        assert!(matches!(
            p.evaluate_batch_into(
                &[C::ZERO],
                &mut [C::ZERO],
                &mut short,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .view(),
            SciOxideErrorView::DimensionMismatch { .. }
        ));
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

#[test]
fn extreme_ranges_subnormals_and_exact_endpoints() {
    fn check<R: RealScalar>() {
        let policy = PolynomialApproximationPolicy::default();
        let p = ApproxPolynomial::bernstein_with_options(
            vec![-R::MAX, R::MAX],
            -R::MAX,
            R::MAX,
            policy,
        )
        .unwrap();
        let evaluation = p.evaluate_with_derivative(R::ZERO).unwrap();
        assert_eq!(evaluation.value(), R::ZERO);
        assert_eq!(evaluation.derivative(), R::ONE);
        let p = ApproxPolynomial::bernstein_with_options(vec![R::MAX; 3], R::ZERO, R::ONE, policy)
            .unwrap();
        let evaluation = p.evaluate_with_derivative(R::HALF).unwrap();
        assert_eq!(evaluation.value(), R::MAX);
        assert_eq!(evaluation.derivative(), R::ZERO);
        let tiny = R::MIN_POSITIVE / R::try_from_f64_lossy(16.).unwrap();
        let p = ApproxPolynomial::bernstein_with_options(
            vec![tiny, R::TWO * tiny, tiny],
            R::ZERO,
            R::ONE,
            policy,
        )
        .unwrap();
        let evaluation = p.evaluate_with_derivative(R::HALF).unwrap();
        let expected = tiny + tiny * R::HALF;
        assert!((evaluation.value() - expected).abs() <= R::MIN_POSITIVE * R::EPSILON);
        assert_eq!(evaluation.derivative(), R::ZERO);
    }
    check::<f32>();
    check::<f64>();
    let p = ApproxPolynomial::bernstein(vec![-0.0_f64, 1., 2., -0.0], -3., 7.).unwrap();
    for x in [-3., 7.] {
        assert_eq!(p.evaluate(x).unwrap().to_bits(), (-0.0_f64).to_bits());
    }
    assert!(p.evaluate_with_derivative(f64::NAN).is_err());
    assert!(p.evaluate_with_derivative(f64::MAX).is_err());
}

#[test]
fn degree_zero_axes_ignore_unrepresentable_normalized_coordinates() {
    let p = ApproxPolynomial::bernstein(vec![3.0_f64], 0., f64::MIN_POSITIVE).unwrap();
    let result = p.evaluate_with_derivative(f64::MAX).unwrap();
    assert_eq!(result.value(), 3.);
    assert_eq!(result.derivative(), 0.);
    let net = TensorBernstein::new(
        &[0, 1],
        vec![
            Interval::new(0., f64::MIN_POSITIVE).unwrap(),
            Interval::new(0., 1.).unwrap(),
        ],
        1,
        vec![2., 4.],
    )
    .unwrap();
    let derivative = net.derivative(0).unwrap();
    let raised = net.elevate_degree(0, 4).unwrap();
    let (left, right) = net.subdivide(0, f64::MIN_POSITIVE / 2.).unwrap();
    for (tensor, expected) in [
        (&net, 3.),
        (&derivative, 0.),
        (&raised, 3.),
        (&left, 3.),
        (&right, 3.),
    ] {
        // The elevated axis is checked on its domain; only a degree-zero axis
        // can skip an otherwise unrepresentable normalization.
        let x = if tensor.extents()[0] == 1 {
            f64::MAX
        } else {
            0.
        };
        let view = tensor.as_view();
        let mut work = vec![0.; view.workspace_requirements(1).unwrap().scratch_len()];
        let mut out = [0.];
        view.evaluate_into(
            &[x, 0.5],
            &mut out,
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(out, [expected]);
    }
}
