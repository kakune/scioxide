//! The scalar forms of `Univariate` against the protocol forms with `[x]` and
//! `Reject`, called with method syntax on every interpolant of one variable.
#![allow(clippy::unwrap_used)]

use scioxide_core::{SciOxideError, SciOxideErrorKind};
use scioxide_interp::{
    CubicSpline, Differentiable, ExtrapolationPolicy, Integrable, Interpolant, Univariate,
};

/// Asserts that two results are identical, bit for bit for the values.
///
/// `Debug` prints the shortest representation that round-trips, so equal
/// strings mean equal bits for every non-NaN component, including signed zeros.
fn same<T: core::fmt::Debug>(left: &Result<T, SciOxideError>, right: &Result<T, SciOxideError>) {
    assert_eq!(format!("{left:?}"), format!("{right:?}"));
}

fn kind<T>(result: Result<T, SciOxideError>) -> SciOxideErrorKind {
    result.err().map(|error| error.kind()).unwrap()
}

/// `value_at` and `values_at` against `evaluate` and `evaluate_many`, the
/// domain and finiteness errors, and an empty or failing batch.
macro_rules! values {
    ($model:expr, $s:ty, $xs:expr, $outside:expr) => {{
        let model = &$model;
        let xs: &[$s] = &$xs;
        let reject = ExtrapolationPolicy::<$s>::Reject;
        for &x in xs {
            same(&model.value_at(x), &model.evaluate([x], reject));
            same(&(&model).value_at(x), &model.evaluate([x], reject));
        }
        let points: Vec<[$s; 1]> = xs.iter().map(|&x| [x]).collect();
        let pointwise: Result<Vec<_>, SciOxideError> =
            xs.iter().map(|&x| model.evaluate([x], reject)).collect();
        same(&model.values_at(xs), &pointwise);
        same(&model.values_at(xs), &model.evaluate_many(&points, reject));
        assert!(model.values_at(&[]).unwrap().is_empty());
        let outside: $s = $outside;
        assert_eq!(
            kind(model.value_at(outside)),
            SciOxideErrorKind::DomainError
        );
        assert_eq!(
            kind(model.value_at(<$s>::NAN)),
            SciOxideErrorKind::NonFinite
        );
        let mut failing = xs.to_vec();
        failing.insert(1, outside);
        assert_eq!(
            kind(model.values_at(&failing)),
            SciOxideErrorKind::DomainError
        );
        failing.insert(1, <$s>::INFINITY);
        assert_eq!(
            kind(model.values_at(&failing)),
            SciOxideErrorKind::NonFinite
        );
    }};
}

/// `derivative_at` against `gradient` with `[x]` and `Reject`.
macro_rules! derivative {
    ($model:expr, $s:ty, $xs:expr, $outside:expr) => {{
        let model = &$model;
        let xs: &[$s] = &$xs;
        for &x in xs {
            // A knot where the derivative is undefined fails in both forms.
            let gradient = model.gradient([x], ExtrapolationPolicy::Reject);
            same(&model.derivative_at(x), &gradient.map(|[slope]| slope));
        }
        assert_eq!(
            kind(model.derivative_at($outside)),
            SciOxideErrorKind::DomainError
        );
    }};
}

/// `integrate` against `integral` with `[lower]`, `[upper]` and `Reject`.
macro_rules! integral {
    ($model:expr, $s:ty, $lower:expr, $upper:expr, $outside:expr) => {{
        let model = &$model;
        let (lower, upper): ($s, $s) = ($lower, $upper);
        let reject = ExtrapolationPolicy::Reject;
        same(
            &model.integrate(lower, upper),
            &model.integral([lower], [upper], reject),
        );
        same(
            &model.integrate(lower, lower),
            &model.integral([lower], [lower], reject),
        );
        assert_eq!(
            kind(model.integrate(upper, lower)),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            kind(model.integrate(lower, $outside)),
            SciOxideErrorKind::DomainError
        );
    }};
}

macro_rules! widths {
    ($module:ident, $s:ty) => {
        mod $module {
            use super::*;
            use scioxide_core::Complex;
            use scioxide_core::Interval;
            use scioxide_interp::{
                AkimaInterpolator, BSpline, BSplineBasis, BarycentricInterpolator,
                BulirschStoerInterpolator, Covariance, CovarianceKernel,
                FloaterHormannInterpolator, GaussianProcessModel, HermiteInterpolator, KrigingMean,
                LocalModelPlan, LocalPolynomialModel, LocalPolynomialOptions, LocalRbfModel,
                LocalRbfOptions, LocalSamples, NearestNeighborInterpolator, NevilleInterpolator,
                NewtonInterpolator, Nurbs, PadeApproximant, PchipInterpolator, RationalDomain,
                RbfKernel, ScatteredValues, SimplexLinearInterpolator, aaa,
            };
            use scioxide_spatial::{Delaunay, PointSet};
            type S = $s;

            fn nodes() -> Vec<S> {
                vec![0.0, 0.5, 1.25, 2.0, 3.0]
            }
            fn real() -> Vec<S> {
                nodes().iter().map(|&t| (t * 1.3).sin() + 0.5 * t).collect()
            }
            fn complex() -> Vec<Complex<S>> {
                nodes()
                    .iter()
                    .map(|&t| Complex::new(t * t, 1.0 - t))
                    .collect()
            }
            fn arrays() -> Vec<[S; 2]> {
                nodes().iter().map(|&t| [t, 2.0 - t * t]).collect()
            }
            const QUERIES: [S; 6] = [0.0, 0.3, 1.0, 1.25, 2.6, 3.0];
            const OUTSIDE: S = 3.5;

            #[test]
            fn piecewise_cubics_and_splines() {
                let x = nodes();
                values!(
                    CubicSpline::new(x.clone(), real()).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
                values!(
                    CubicSpline::new(x.clone(), complex()).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
                values!(
                    CubicSpline::new(x.clone(), arrays()).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
                let cubic = CubicSpline::new(x.clone(), complex()).unwrap();
                derivative!(cubic, S, QUERIES, OUTSIDE);
                integral!(cubic, S, 0.25, 2.75, OUTSIDE);
                let pchip = PchipInterpolator::new(x.clone(), real()).unwrap();
                values!(pchip, S, QUERIES, OUTSIDE);
                derivative!(pchip, S, QUERIES, OUTSIDE);
                integral!(pchip, S, 0.0, 3.0, OUTSIDE);
                let akima = AkimaInterpolator::new(x.clone(), arrays()).unwrap();
                values!(akima, S, QUERIES, OUTSIDE);
                derivative!(akima, S, QUERIES, OUTSIDE);
                integral!(akima, S, 0.5, 1.5, OUTSIDE);
                let line = BSpline::linear(x.clone(), complex()).unwrap();
                values!(line, S, QUERIES, OUTSIDE);
                derivative!(line, S, QUERIES, OUTSIDE);
                integral!(line, S, 0.1, 2.9, OUTSIDE);
                let basis = BSplineBasis::clamped(&[0.0, 1.0, 3.0], 2).unwrap();
                let nurbs =
                    Nurbs::new([basis], arrays()[..4].to_vec(), vec![1.0, 2.0, 0.5, 1.0]).unwrap();
                values!(nurbs, S, QUERIES, OUTSIDE);
                derivative!(nurbs, S, QUERIES, OUTSIDE);
            }

            #[test]
            fn polynomials_and_rational_interpolants() {
                let x = nodes();
                let barycentric = BarycentricInterpolator::new(x.clone(), complex()).unwrap();
                values!(barycentric, S, QUERIES, OUTSIDE);
                derivative!(barycentric, S, QUERIES, OUTSIDE);
                let newton = NewtonInterpolator::new(x.clone(), arrays()).unwrap();
                values!(newton, S, QUERIES, OUTSIDE);
                derivative!(newton, S, QUERIES, OUTSIDE);
                let slopes: Vec<S> = x.iter().map(|&t| (t * 1.3).cos() * 1.3 + 0.5).collect();
                let hermite = HermiteInterpolator::new(x.clone(), real(), slopes).unwrap();
                values!(hermite, S, QUERIES, OUTSIDE);
                derivative!(hermite, S, QUERIES, OUTSIDE);
                values!(
                    NevilleInterpolator::new(x.clone(), real()).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
                values!(
                    BulirschStoerInterpolator::new(x.clone(), real()).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
                values!(
                    FloaterHormannInterpolator::new(x, complex(), 2).unwrap(),
                    S,
                    QUERIES,
                    OUTSIDE
                );
            }

            #[test]
            fn rational_approximants_in_real_and_complex_arithmetic() {
                let x = nodes();
                let f = |t: S| 1.0 / (1.0 + t * t);
                let y: Vec<S> = x.iter().map(|&t| f(t)).collect();
                values!(aaa(&x, &y).unwrap(), S, QUERIES, OUTSIDE);
                let z: Vec<Complex<S>> = x.iter().map(|&t| Complex::new(t, 0.0)).collect();
                let w: Vec<Complex<S>> = x.iter().map(|&t| Complex::new(f(t), t)).collect();
                values!(aaa(&z, &w).unwrap(), S, QUERIES, OUTSIDE);
                // exp(t) about 0: the [2/2] Padé approximant on [0, 1].
                let taylor: [S; 5] = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
                let domain = RationalDomain::Real(Interval::new(0.0, 1.0).unwrap());
                let pade = PadeApproximant::new(&taylor, 2, 2, 0.0, domain).unwrap();
                values!(pade, S, [0.0, 0.25, 0.5, 1.0], 1.5);
                let taylor = taylor.map(|c| Complex::new(c, -c));
                let pade =
                    PadeApproximant::new(&taylor, 2, 2, Complex::new(0.0, 0.0), domain).unwrap();
                values!(pade, S, [0.0, 0.25, 0.5, 1.0], 1.5);
            }

            #[test]
            fn scattered_models_of_one_variable() {
                let points: Vec<[S; 1]> = nodes().iter().map(|&t| [t]).collect();
                let mesh = Delaunay::new(&PointSet::from_points(&points).unwrap()).unwrap();
                let values = ScatteredValues::from_input_values(&mesh, complex()).unwrap();
                let linear = SimplexLinearInterpolator::<S, 1, _>::new(&mesh, &values).unwrap();
                values!(linear, S, QUERIES, OUTSIDE);
                derivative!(linear, S, [0.3, 0.75, 1.6, 2.5], OUTSIDE);
                let nearest = NearestNeighborInterpolator::<S, 1, _>::new(&mesh, &values).unwrap();
                values!(nearest, S, QUERIES, OUTSIDE);
            }

            #[test]
            fn local_models_of_one_variable() {
                let points: Vec<[S; 1]> = nodes().iter().map(|&t| [t]).collect();
                let samples = LocalSamples::new(&points, &real()).unwrap();
                let sites = samples.points().unwrap();
                let plan = LocalModelPlan::<S, 1>::new(&sites, samples.values().unwrap()).unwrap();
                let kernel = RbfKernel::InverseMultiquadric(1.0);
                let options = LocalRbfOptions::new(5, kernel).unwrap();
                let rbf = LocalRbfModel::new(&plan, [1.0], options).unwrap();
                let queries = [0.0, 0.5, 1.7, 3.0];
                values!(rbf, S, queries, OUTSIDE);
                let options = LocalPolynomialOptions::new(2, 3, 5);
                let polynomial = LocalPolynomialModel::new(&plan, [1.0], options).unwrap();
                values!(polynomial, S, queries, OUTSIDE);
                let covariance =
                    Covariance::new(CovarianceKernel::SquaredExponential, 1.0, [0.75]).unwrap();
                let kriging = GaussianProcessModel::new(
                    &sites,
                    samples.values().unwrap(),
                    covariance,
                    &[1e-3; 5],
                    KrigingMean::Ordinary,
                )
                .unwrap();
                let prediction = kriging.prediction(0.0).unwrap();
                values!(prediction, S, queries, OUTSIDE);
            }
        }
    };
}

widths!(single, f32);
widths!(double, f64);

#[test]
fn scalar_cubic_spline_forms_execute() {
    let x = vec![0.0_f64, 1.0, 2.5, 3.0, 4.0];
    let y = vec![1.0, -1.0, 0.5, 2.0, 0.0];
    let spline = CubicSpline::new(x, y).unwrap();
    let mut sum = 0.0;
    {
        sum += spline.value_at(1.75).unwrap();
        sum += spline.derivative_at(1.75).unwrap();
        sum += spline.integrate(0.5, 3.5).unwrap();
    };
    assert!(sum.is_finite());
}
