//! Sample constructors against `with_options` with `SampleOptions::default()`.

use scioxide_core::{
    InvalidInputReason, SciOxideError, SciOxideErrorKind, SciOxideErrorView, Tolerance,
};
use scioxide_interp::{
    AkimaInterpolator, AkimaMethod, BSpline, BarycentricInterpolator, BicubicInterpolator,
    BilinearInterpolator, BulirschStoerInterpolator, CubicSpline, Differentiable,
    ExtrapolationPolicy, FloaterHormannInterpolator, HermiteInterpolator, Interpolant,
    NevilleInterpolator, NewtonInterpolator, NodeSeparation, PchipInterpolator, SampleOptions,
    SplineBoundary,
};

use crate::support::assert_error;

/// Returns the argument of an `InvalidInput` error, or `None` for any other outcome.
fn blamed<T>(result: &Result<T, SciOxideError>) -> Option<&'static str> {
    result.as_ref().err().and_then(|error| match error.view() {
        SciOxideErrorView::InvalidInput { argument, .. } => Some(argument.as_str()),
        _ => None,
    })
}

/// Asserts `InvalidInput` (`Inconsistent`) blaming `argument`.
fn assert_inconsistent<T>(result: &Result<T, SciOxideError>, argument: &str) {
    let error = assert_error(result, SciOxideErrorKind::InvalidInput);
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::Inconsistent,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(blamed(result), Some(argument));
}

macro_rules! sample_constructors {
    ($module:ident, $S:ty) => {
        mod $module {
            use super::*;

            type S = $S;
            const REJECT: ExtrapolationPolicy<S> = ExtrapolationPolicy::Reject;
            const QUERIES: [S; 5] = [0.0, 0.35, 1.9, 2.95, 4.1];
            const POINTS: [[S; 2]; 5] = [
                [0.0, -1.0],
                [0.35, 0.2],
                [1.9, 0.5],
                [0.7, -0.4],
                [2.2, 0.0],
            ];

            fn x() -> Vec<S> {
                vec![0.0, 0.7, 1.5, 2.2, 3.0, 4.1]
            }
            fn y() -> Vec<S> {
                vec![1.0, -0.4, 0.8, 2.5, 1.2, -1.0]
            }
            fn dy() -> Vec<S> {
                vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5]
            }
            /// A 4 × 3 grid with `z[j * nx + i]` at `(gx[i], gy[j])`.
            fn grid() -> (Vec<S>, Vec<S>, Vec<S>) {
                let z = (0..12_u8).map(|k| (S::from(k) * 0.37).sin()).collect();
                (vec![0.0, 0.7, 1.5, 2.2], vec![-1.0, 0.0, 0.5], z)
            }
            fn defaults() -> SampleOptions<S> {
                SampleOptions::default()
            }
            fn bits(values: &[S]) -> Vec<u64> {
                values.iter().map(|&v| f64::from(v).to_bits()).collect()
            }
            /// Asserts equal values at the five queries, bit for bit.
            fn same_curve<I: Interpolant<S, 1, Value = S>>(left: &I, right: &I) {
                for q in QUERIES {
                    let (a, b) = (left.evaluate([q], REJECT), right.evaluate([q], REJECT));
                    assert_eq!(
                        a.map(f64::from).map(f64::to_bits),
                        b.map(f64::from).map(f64::to_bits)
                    );
                }
            }
            /// Asserts equal values at the five grid points, bit for bit.
            fn same_surface<I: Interpolant<S, 2, Value = S>>(left: &I, right: &I) {
                for p in POINTS {
                    let (a, b) = (left.evaluate(p, REJECT), right.evaluate(p, REJECT));
                    assert_eq!(
                        a.map(f64::from).map(f64::to_bits),
                        b.map(f64::from).map(f64::to_bits)
                    );
                }
            }

            #[test]
            fn defaults_are_exact() {
                assert_eq!(defaults(), SampleOptions::new());
                assert_eq!(defaults().node_separation(), NodeSeparation::Exact);
                let tolerant = NodeSeparation::Tolerant(Tolerance::new(0.5, 0.0).unwrap());
                let options = SampleOptions::<S>::new().with_node_separation(tolerant);
                assert_eq!(options.node_separation(), tolerant);
            }

            #[test]
            fn piecewise_new_equals_with_options() {
                let cubic = CubicSpline::new(x(), y()).unwrap();
                let not_a_knot = SplineBoundary::NotAKnot;
                let configured =
                    CubicSpline::with_options(x(), y(), not_a_knot, defaults()).unwrap();
                assert_eq!(bits(cubic.slopes()), bits(configured.slopes()));
                same_curve(&cubic, &configured);

                let pchip = PchipInterpolator::new(x(), y()).unwrap();
                let configured = PchipInterpolator::with_options(x(), y(), defaults()).unwrap();
                assert_eq!(bits(pchip.slopes()), bits(configured.slopes()));
                same_curve(&pchip, &configured);

                let akima = AkimaInterpolator::new(x(), y()).unwrap();
                let configured =
                    AkimaInterpolator::with_options(x(), y(), AkimaMethod::Akima, defaults())
                        .unwrap();
                assert_eq!(bits(akima.slopes()), bits(configured.slopes()));
                same_curve(&akima, &configured);
            }

            #[test]
            fn named_choices_equal_with_options_with_the_same_value() {
                let (first, last) = (0.5, -2.0);
                for boundary in [
                    SplineBoundary::Natural,
                    SplineBoundary::Clamped(first, last),
                    SplineBoundary::NotAKnot,
                    SplineBoundary::Periodic,
                ] {
                    // Periodic ends need equal end values.
                    let mut values = y();
                    values[5] = values[0];
                    let named = CubicSpline::with_boundary(x(), values.clone(), boundary).unwrap();
                    let configured =
                        CubicSpline::with_options(x(), values, boundary, defaults()).unwrap();
                    assert_eq!(bits(named.slopes()), bits(configured.slopes()));
                    same_curve(&named, &configured);
                }
                for method in [AkimaMethod::Akima, AkimaMethod::Makima] {
                    let named = AkimaInterpolator::with_method(x(), y(), method).unwrap();
                    let configured =
                        AkimaInterpolator::with_options(x(), y(), method, defaults()).unwrap();
                    assert_eq!(bits(named.slopes()), bits(configured.slopes()));
                    same_curve(&named, &configured);
                }
                // The documented choices of `new`.
                let natural =
                    CubicSpline::with_boundary(x(), y(), SplineBoundary::Natural).unwrap();
                let not_a_knot =
                    CubicSpline::with_boundary(x(), y(), SplineBoundary::NotAKnot).unwrap();
                let cubic = CubicSpline::new(x(), y()).unwrap();
                assert_eq!(bits(cubic.slopes()), bits(not_a_knot.slopes()));
                assert_ne!(bits(cubic.slopes()), bits(natural.slopes()));
                let makima = AkimaInterpolator::with_method(x(), y(), AkimaMethod::Makima).unwrap();
                let akima = AkimaInterpolator::new(x(), y()).unwrap();
                assert_ne!(bits(akima.slopes()), bits(makima.slopes()));
            }

            #[test]
            fn polynomial_new_equals_with_options() {
                let newton = NewtonInterpolator::new(x(), y()).unwrap();
                let configured = NewtonInterpolator::with_options(x(), y(), defaults()).unwrap();
                assert_eq!(bits(newton.coefficients()), bits(configured.coefficients()));
                same_curve(&newton, &configured);

                let neville = NevilleInterpolator::new(x(), y()).unwrap();
                let configured = NevilleInterpolator::with_options(x(), y(), defaults()).unwrap();
                assert_eq!(bits(neville.values()), bits(configured.values()));
                same_curve(&neville, &configured);

                let rational = BulirschStoerInterpolator::new(x(), y()).unwrap();
                let configured =
                    BulirschStoerInterpolator::with_options(x(), y(), defaults()).unwrap();
                assert_eq!(bits(rational.values()), bits(configured.values()));
                same_curve(&rational, &configured);

                let barycentric = BarycentricInterpolator::new(x(), y()).unwrap();
                let configured =
                    BarycentricInterpolator::with_options(x(), y(), defaults()).unwrap();
                assert_eq!(bits(barycentric.weights()), bits(configured.weights()));
                same_curve(&barycentric, &configured);

                let hermite = HermiteInterpolator::new(x(), y(), dy()).unwrap();
                let configured =
                    HermiteInterpolator::with_options(x(), y(), dy(), defaults()).unwrap();
                same_curve(&hermite, &configured);

                let blend = FloaterHormannInterpolator::new(x(), y(), 2).unwrap();
                let configured =
                    FloaterHormannInterpolator::with_options(x(), y(), 2, defaults()).unwrap();
                assert_eq!(bits(blend.weights()), bits(configured.weights()));
                same_curve(&blend, &configured);
            }

            #[test]
            fn grid_new_equals_with_options() {
                let (gx, gy, z) = grid();
                let bilinear =
                    BilinearInterpolator::new(gx.clone(), gy.clone(), z.clone()).unwrap();
                let configured = BilinearInterpolator::with_options(
                    gx.clone(),
                    gy.clone(),
                    z.clone(),
                    defaults(),
                )
                .unwrap();
                same_surface(&bilinear, &configured);

                let bicubic = BicubicInterpolator::new(gx.clone(), gy.clone(), z.clone()).unwrap();
                let configured = BicubicInterpolator::with_options(gx, gy, z, defaults()).unwrap();
                same_surface(&bicubic, &configured);
                for p in POINTS {
                    let (a, b) = (
                        bicubic.gradient(p, REJECT).unwrap(),
                        configured.gradient(p, REJECT).unwrap(),
                    );
                    assert_eq!(bits(&a), bits(&b));
                }
            }

            #[test]
            fn tolerant_options_reject_a_gap_that_new_accepts() {
                // The gap 2⁻¹⁰ is below the absolute tolerance 2⁻⁸.
                let close: Vec<S> = vec![0.0, 1.0, 1.0 + 1.0 / 1024.0, 2.0, 3.0];
                let values: Vec<S> = vec![1.0, 0.5, 0.25, -1.0, 2.0];
                let tolerant = NodeSeparation::Tolerant(Tolerance::new(1.0 / 256.0, 0.0).unwrap());
                let options = SampleOptions::new().with_node_separation(tolerant);
                let (c, v) = (|| close.clone(), || values.clone());
                let boundary = SplineBoundary::NotAKnot;

                assert!(CubicSpline::new(c(), v()).is_ok());
                assert_inconsistent(&CubicSpline::with_options(c(), v(), boundary, options), "x");
                assert!(PchipInterpolator::new(c(), v()).is_ok());
                assert_inconsistent(&PchipInterpolator::with_options(c(), v(), options), "x");
                assert!(AkimaInterpolator::new(c(), v()).is_ok());
                let method = AkimaMethod::Makima;
                assert_inconsistent(
                    &AkimaInterpolator::with_options(c(), v(), method, options),
                    "x",
                );
                assert!(HermiteInterpolator::new(c(), v(), v()).is_ok());
                assert_inconsistent(
                    &HermiteInterpolator::with_options(c(), v(), v(), options),
                    "x",
                );
                assert!(NewtonInterpolator::new(c(), v()).is_ok());
                assert_inconsistent(&NewtonInterpolator::with_options(c(), v(), options), "x");
                assert!(NevilleInterpolator::new(c(), v()).is_ok());
                assert_inconsistent(&NevilleInterpolator::with_options(c(), v(), options), "x");
                assert!(BulirschStoerInterpolator::new(c(), v()).is_ok());
                assert_inconsistent(
                    &BulirschStoerInterpolator::with_options(c(), v(), options),
                    "x",
                );
                assert!(BarycentricInterpolator::new(c(), v()).is_ok());
                assert_inconsistent(
                    &BarycentricInterpolator::with_options(c(), v(), options),
                    "x",
                );
                assert!(FloaterHormannInterpolator::new(c(), v(), 2).is_ok());
                assert_inconsistent(
                    &FloaterHormannInterpolator::with_options(c(), v(), 2, options),
                    "x",
                );

                let (gy, z) = (vec![0.0, 1.0], [values.clone(), values.clone()].concat());
                assert!(BilinearInterpolator::new(c(), gy.clone(), z.clone()).is_ok());
                let bilinear =
                    BilinearInterpolator::with_options(c(), gy.clone(), z.clone(), options);
                assert_inconsistent(&bilinear, "x");
                assert!(BicubicInterpolator::new(c(), gy.clone(), z.clone()).is_ok());
                assert_inconsistent(&BicubicInterpolator::with_options(c(), gy, z, options), "x");
            }

            #[test]
            fn structured_nodes_span_lower_to_upper() {
                let values: Vec<S> = vec![1.0, -0.5, 0.25, 2.0, 0.0];
                let (lower, upper): (S, S) = (-2.0, 3.0);
                let tolerance = 8.0 * f64::from(S::EPSILON);
                let pi = std::f64::consts::PI;
                // Unit nodes on [−1, 1] in increasing order, before the affine map.
                let equispaced = [-1.0, -0.5, 0.0, 0.5, 1.0];
                let roots = [9.0, 7.0, 5.0, 3.0, 1.0].map(|k: f64| (k * pi / 10.0).cos());
                let lobatto = [4.0, 3.0, 2.0, 1.0, 0.0].map(|k: f64| (k * pi / 4.0).cos());
                type Constructor =
                    fn(Vec<S>, S, S) -> Result<BarycentricInterpolator<S>, SciOxideError>;
                let families: [(Constructor, [f64; 5]); 3] = [
                    (BarycentricInterpolator::from_equispaced, equispaced),
                    (BarycentricInterpolator::from_chebyshev_roots, roots),
                    (BarycentricInterpolator::from_chebyshev_lobatto, lobatto),
                ];
                for (constructor, unit) in families {
                    let structured = constructor(values.clone(), lower, upper).unwrap();
                    assert_eq!(bits(structured.values()), bits(&values));
                    for (&node, t) in structured.nodes().iter().zip(unit) {
                        let expected = 0.5 + 2.5 * t;
                        assert!(
                            (f64::from(node) - expected).abs() <= tolerance * 3.0,
                            "{node} {expected}"
                        );
                    }
                    // The same polynomial as the O(n²) weights on the generated nodes.
                    let general =
                        BarycentricInterpolator::new(structured.nodes().to_vec(), values.clone())
                            .unwrap();
                    let [domain] = structured.domains();
                    for fraction in [0.0, 0.2, 0.45, 0.8, 1.0] {
                        let q = domain.lower() + fraction * (domain.upper() - domain.lower());
                        let q = q.clamp(domain.lower(), domain.upper());
                        let a = f64::from(structured.evaluate([q], REJECT).unwrap());
                        let b = f64::from(general.evaluate([q], REJECT).unwrap());
                        assert!(
                            (a - b).abs() <= 64.0 * tolerance * b.abs().max(1.0),
                            "{q}: {a} {b}"
                        );
                    }
                }
                // Endpoint families keep the bounds verbatim.
                let ends =
                    BarycentricInterpolator::from_chebyshev_lobatto(values.clone(), lower, upper)
                        .unwrap();
                assert_eq!((ends.nodes()[0], ends.nodes()[4]), (lower, upper));
                let ends = BarycentricInterpolator::from_equispaced(values, lower, upper).unwrap();
                assert_eq!((ends.nodes()[0], ends.nodes()[4]), (lower, upper));
                // One value sits at the midpoint, also of a point interval.
                let single =
                    BarycentricInterpolator::from_chebyshev_roots(vec![7.0], lower, upper).unwrap();
                assert_eq!(single.nodes(), [0.5]);
                let point = BarycentricInterpolator::from_equispaced(vec![7.0], 5.0, 5.0).unwrap();
                assert_eq!(point.nodes(), [5.0]);
            }

            #[test]
            fn structured_constructors_reject_bad_bounds_and_values() {
                type Constructor =
                    fn(Vec<S>, S, S) -> Result<BarycentricInterpolator<S>, SciOxideError>;
                let constructors: [Constructor; 3] = [
                    BarycentricInterpolator::from_equispaced,
                    BarycentricInterpolator::from_chebyshev_roots,
                    BarycentricInterpolator::from_chebyshev_lobatto,
                ];
                for constructor in constructors {
                    assert_inconsistent(&constructor(vec![1.0, 2.0], 1.0, -1.0), "lower");
                    assert_inconsistent(&constructor(vec![1.0, 2.0], 2.0, 2.0), "upper");
                    for bad in [S::NAN, S::INFINITY] {
                        assert_error(
                            &constructor(vec![1.0, 2.0], bad, 1.0),
                            SciOxideErrorKind::NonFinite,
                        );
                        assert_error(
                            &constructor(vec![1.0, 2.0], -1.0, bad),
                            SciOxideErrorKind::NonFinite,
                        );
                        assert_error(
                            &constructor(vec![1.0, bad], -1.0, 1.0),
                            SciOxideErrorKind::NonFinite,
                        );
                    }
                    assert_error(
                        &constructor(Vec::new(), -1.0, 1.0),
                        SciOxideErrorKind::InvalidInput,
                    );
                }
            }

            /// Checks the shared validation of one constructor of `(x, y)`.
            fn check_errors<T>(
                minimum: usize,
                build: impl Fn(Vec<S>, Vec<S>) -> Result<T, SciOxideError>,
            ) {
                let one = build(vec![1.0], vec![2.0]);
                if minimum == 1 {
                    assert!(one.is_ok());
                } else {
                    assert_error(&one, SciOxideErrorKind::InvalidInput);
                    assert_eq!(blamed(&one), Some("x"));
                }
                assert_inconsistent(&build(vec![0.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]), "x");
                assert_inconsistent(&build(vec![0.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]), "x");
                assert_error(
                    &build(vec![0.0, 1.0, 2.0], vec![1.0, 2.0]),
                    SciOxideErrorKind::DimensionMismatch,
                );
                assert_error(
                    &build(vec![0.0, S::NAN, 2.0], vec![1.0, 2.0, 3.0]),
                    SciOxideErrorKind::NonFinite,
                );
                assert_error(
                    &build(vec![0.0, 1.0, 2.0], vec![1.0, S::NAN, 3.0]),
                    SciOxideErrorKind::NonFinite,
                );
            }

            #[test]
            fn constructors_reject_invalid_samples() {
                check_errors(2, CubicSpline::new);
                check_errors(2, |x, y| {
                    CubicSpline::with_boundary(x, y, SplineBoundary::Natural)
                });
                check_errors(2, PchipInterpolator::new);
                check_errors(2, AkimaInterpolator::new);
                check_errors(2, |x, y| {
                    AkimaInterpolator::with_method(x, y, AkimaMethod::Makima)
                });
                check_errors(2, BSpline::linear);
                check_errors(1, |x, y: Vec<S>| HermiteInterpolator::new(x, y.clone(), y));
                check_errors(1, NewtonInterpolator::new);
                check_errors(1, NevilleInterpolator::new);
                check_errors(1, BulirschStoerInterpolator::new);
                check_errors(1, BarycentricInterpolator::new);
                check_errors(1, |x, y| FloaterHormannInterpolator::new(x, y, 0));
                // Grids: `x` varies, the second axis is [0, 1].
                let grid = |y: Vec<S>| [y.clone(), y].concat();
                check_errors(2, |x, y| {
                    BilinearInterpolator::new(x, vec![0.0, 1.0], grid(y))
                });
                check_errors(2, |x, y| {
                    BicubicInterpolator::new(x, vec![0.0, 1.0], grid(y))
                });
                check_errors(2, |x, y| {
                    BicubicInterpolator::monotone(x, vec![0.0, 1.0], grid(y))
                });
            }
        }
    };
}

sample_constructors!(single, f32);
sample_constructors!(double, f64);
