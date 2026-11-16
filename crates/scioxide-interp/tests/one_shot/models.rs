//! `new` of the models whose options have a `Default` against `with_options`.

use scioxide_core::{Complex, Interval, RealScalar};
use scioxide_interp::{
    CloughTocherInterpolator, CloughTocherOptions, Covariance, CovarianceKernel,
    ExtrapolationPolicy, GaussianProcessModel, Interpolant, KrigingMean, LocalModelOptions,
    LocalSamples, NaturalNeighborInterpolator, NearestNeighborInterpolator, PadeApproximant,
    RationalDomain, ScatteredOptions, ScatteredValues, SimplexLinearInterpolator, Univariate,
};
use scioxide_poly::PolynomialApproximationPolicy;
use scioxide_spatial::{Delaunay, PointSet};

use crate::support::from_f64;

/// Five query points inside the unit square.
const QUERIES: [[f64; 2]; 5] = [
    [0.1, 0.2],
    [0.5, 0.5],
    [0.33, 0.61],
    [0.9, 0.05],
    [0.7, 0.8],
];

fn bits<S: RealScalar>(values: impl IntoIterator<Item = S>) -> Vec<u64> {
    values
        .into_iter()
        .map(|value| value.try_to_f64_exact().unwrap().to_bits())
        .collect()
}

fn queries<S: RealScalar>() -> Vec<[S; 2]> {
    QUERIES.map(|[x, y]| [from_f64(x), from_f64(y)]).to_vec()
}

/// The values of a model at the query points.
fn values_at<S: RealScalar, M: Interpolant<S, 2, Value = S>>(model: &M) -> Vec<u64> {
    let points = queries::<S>();
    bits(
        points
            .iter()
            .map(|&point| model.evaluate(point, ExtrapolationPolicy::Reject).unwrap()),
    )
}

/// The unit-square corners and four interior sites with `f(x, y) = e^x cos 2y + xy`.
fn scattered<S: RealScalar>() -> (Delaunay<S>, Vec<S>) {
    let sites: Vec<[S; 2]> = [
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.],
        [0.4, 0.35],
        [0.7, 0.6],
        [0.25, 0.8],
        [0.8, 0.2],
    ]
    .map(|[x, y]| [from_f64(x), from_f64(y)])
    .to_vec();
    let mesh = Delaunay::new(&PointSet::from_points(&sites).unwrap()).unwrap();
    let heights = sites
        .iter()
        .map(|&[x, y]| x.exp() * (y + y).cos() + x * y)
        .collect();
    (mesh, heights)
}

fn scattered_family<S: RealScalar>() {
    let (mesh, heights) = scattered::<S>();
    let values = ScatteredValues::from_input_values(&mesh, heights).unwrap();
    let options = ScatteredOptions::default();
    let linear = SimplexLinearInterpolator::<S, 2>::new(&mesh, &values).unwrap();
    let configured = SimplexLinearInterpolator::<S, 2>::with_options(&mesh, &values, options);
    assert_eq!(values_at(&linear), values_at(&configured.unwrap()));
    let nearest = NearestNeighborInterpolator::<S, 2>::new(&mesh, &values).unwrap();
    let configured = NearestNeighborInterpolator::<S, 2>::with_options(&mesh, &values, options);
    assert_eq!(values_at(&nearest), values_at(&configured.unwrap()));
    let natural = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
    let configured = NaturalNeighborInterpolator::with_options(&mesh, &values, options);
    assert_eq!(values_at(&natural), values_at(&configured.unwrap()));
    let clough = CloughTocherInterpolator::new(&mesh, &values).unwrap();
    let configured =
        CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::default())
            .unwrap();
    assert!(clough.gradient_report().termination().is_success());
    assert_eq!(clough.gradient_report(), configured.gradient_report());
    assert_eq!(values_at(&clough), values_at(&configured));
}

#[test]
fn scattered_models_equal_their_default_configured_forms() {
    scattered_family::<f32>();
    scattered_family::<f64>();
}

fn kriging_family<S: RealScalar + scioxide_core::NumericScalar<Real = S>>() {
    let sites: Vec<[S; 2]> = [
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.],
        [0.4, 0.35],
        [0.7, 0.6],
    ]
    .map(|[x, y]| [from_f64(x), from_f64(y)])
    .to_vec();
    let heights: Vec<S> = sites.iter().map(|&[x, y]| x.sin() + y * y).collect();
    let samples = LocalSamples::new(&sites, &heights).unwrap();
    let points = samples.points().unwrap();
    let one = from_f64::<S>(0.6);
    let covariance = Covariance::new(CovarianceKernel::SquaredExponential, S::ONE, [one; 2]);
    let covariance = covariance.unwrap();
    let noise = [from_f64::<S>(1e-3); 6];
    for mean in [KrigingMean::Ordinary, KrigingMean::Constant(&[S::ONE])] {
        let model =
            GaussianProcessModel::new(&points, samples.values().unwrap(), covariance, &noise, mean)
                .unwrap();
        let configured = GaussianProcessModel::with_options(
            &points,
            samples.values().unwrap(),
            covariance,
            &noise,
            mean,
            LocalModelOptions::default(),
        )
        .unwrap();
        assert_eq!(model.fit_report(), configured.fit_report());
        let (model, configured) = (model.prediction(S::ZERO), configured.prediction(S::ZERO));
        let (model, configured) = (model.unwrap(), configured.unwrap());
        for point in queries::<S>() {
            let value = model.evaluate(point, ExtrapolationPolicy::Reject).unwrap();
            let expected = configured
                .evaluate(point, ExtrapolationPolicy::Reject)
                .unwrap();
            assert_eq!(bits(value), bits(expected));
        }
    }
}

#[test]
fn kriging_equals_its_default_configured_form() {
    kriging_family::<f32>();
    kriging_family::<f64>();
}

fn pade_family<S: RealScalar + scioxide_core::NumericScalar<Real = S>>() {
    // Taylor coefficients of log(1 + z) / z at 0.
    let taylor: Vec<S> = (0..9)
        .map(|k| from_f64::<S>(if k % 2 == 0 { 1. } else { -1. } / (k + 1) as f64))
        .collect();
    let domain = RationalDomain::Real(Interval::new(from_f64(-0.5), from_f64(0.9)).unwrap());
    let center = S::ZERO;
    let approximant = PadeApproximant::new(&taylor, 4, 4, center, domain).unwrap();
    let policy = PolynomialApproximationPolicy::default();
    let configured = PadeApproximant::with_options(&taylor, 4, 4, center, domain, policy);
    let configured = configured.unwrap();
    assert_eq!(approximant.fit_report(), configured.fit_report());
    assert_eq!(
        approximant.denominator_rank(),
        configured.denominator_rank()
    );
    let points: Vec<S> = [-0.4, -0.1, 0.2, 0.5, 0.85].map(from_f64).to_vec();
    let values = |model: &PadeApproximant<S>| bits(model.values_at(&points).unwrap());
    assert_eq!(values(&approximant), values(&configured));
    assert!(
        (approximant.value_at(from_f64(0.5)).unwrap() - from_f64(1.5_f64.ln() / 0.5)).abs()
            < from_f64(1e-5)
    );
}

#[test]
fn pade_equals_its_default_configured_form() {
    pade_family::<f32>();
    pade_family::<f64>();
    let taylor: Vec<Complex<f64>> = (0..5)
        .map(|k| Complex::new(1. / f64::from(k + 1), 0.5))
        .collect();
    let domain = RationalDomain::Complex {
        real: Interval::new(-0.5, 0.5).unwrap(),
        imaginary: Interval::new(-0.5, 0.5).unwrap(),
    };
    let zero = Complex::new(0., 0.);
    let approximant = PadeApproximant::new(&taylor, 2, 2, zero, domain).unwrap();
    let policy = PolynomialApproximationPolicy::default();
    let configured = PadeApproximant::with_options(&taylor, 2, 2, zero, domain, policy).unwrap();
    for [re, im] in QUERIES.map(|[x, y]| [x - 0.5, y - 0.5]) {
        let value =
            Interpolant::<f64, 2>::evaluate(&approximant, [re, im], ExtrapolationPolicy::Reject);
        let expected =
            Interpolant::<f64, 2>::evaluate(&configured, [re, im], ExtrapolationPolicy::Reject);
        let (value, expected) = (value.unwrap(), expected.unwrap());
        assert_eq!(bits([value.re, value.im]), bits([expected.re, expected.im]));
    }
}
