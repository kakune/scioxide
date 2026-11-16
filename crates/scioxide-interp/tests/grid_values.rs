//! Grid families accept every affine value and equal per-component scalar evaluation.
#![allow(clippy::unwrap_used)]

use scioxide_core::{AffineValue, Complex, RealScalar, SciOxideErrorKind};
use scioxide_interp::{
    BicubicInterpolator, BilinearInterpolator, Differentiable, ExtrapolationPolicy, Interpolant,
    TwiceDifferentiable,
};

fn empty_grid<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(value: V) {
    for build in [BicubicInterpolator::new, BicubicInterpolator::monotone] {
        let model = build(
            vec![S::ZERO, S::ONE],
            vec![S::ZERO, S::ONE],
            vec![value.clone(); 4],
        )
        .unwrap();
        let point = [S::HALF; 2];
        let reject = ExtrapolationPolicy::Reject;
        assert_eq!(
            model.evaluate(point, reject).unwrap().dimension().unwrap(),
            0
        );
        assert!(
            model
                .gradient(point, reject)
                .unwrap()
                .iter()
                .all(|v| v.dimension().unwrap() == 0)
        );
        assert!(
            model
                .hessian(point, reject)
                .unwrap()
                .iter()
                .flatten()
                .all(|v| v.dimension().unwrap() == 0)
        );
        let mut workspace = model.workspace().unwrap();
        let mut out: [S; 0] = [];
        {
            model
                .evaluate_into(point, reject, &mut out[..], &mut workspace)
                .unwrap();
            model
                .gradient_into(point, reject, &mut out[..], &mut workspace)
                .unwrap();
            model
                .hessian_into(point, reject, &mut out[..], &mut workspace)
                .unwrap();
        };
        for bad in [[S::NAN, S::HALF], [S::TWO, S::HALF]] {
            assert!(
                model
                    .evaluate_into(bad, reject, &mut out[..], &mut workspace)
                    .is_err()
            );
            assert!(
                model
                    .gradient_into(bad, reject, &mut out[..], &mut workspace)
                    .is_err()
            );
            assert!(
                model
                    .hessian_into(bad, reject, &mut out[..], &mut workspace)
                    .is_err()
            );
        }
    }
}

#[test]
fn empty_arrays_and_vectors_preserve_bicubic_query_contracts() {
    empty_grid::<f32, _>([0.0_f32; 0]);
    empty_grid::<f64, _>([0.0_f64; 0]);
    empty_grid::<f32, _>(Vec::<f32>::new());
    empty_grid::<f64, _>(Vec::<f64>::new());
}

fn grid<S: RealScalar>() -> (Vec<S>, Vec<S>) {
    let cv = |v: f64| S::try_from_f64_lossy(v).unwrap();
    (
        [-1., 0., 0.5, 2.].map(cv).to_vec(),
        [-2., 0., 1.].map(cv).to_vec(),
    )
}

/// Three independent components per node, each monotone along both axes so the
/// monotone constructor accepts them.
fn components<S: RealScalar>(x: &[S], y: &[S]) -> [Vec<S>; 3] {
    let cv = |v: f64| S::try_from_f64_lossy(v).unwrap();
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for &b in y {
        for &a in x {
            out[0].push(a * a * a + b);
            out[1].push(cv(3.) * a + b * b * b + cv(0.25));
            out[2].push(-a - cv(2.) * b);
        }
    }
    out
}

fn points<S: RealScalar>() -> [[S; 2]; 4] {
    let cv = |v: f64| S::try_from_f64_lossy(v).unwrap();
    [[0.25, 0.5], [-1.0, -2.0], [1.75, 0.9], [3.0, -0.5]].map(|p| p.map(cv))
}

macro_rules! check_family {
    ($S:ty, $build:path, $policy:expr) => {{
        let (x, y) = grid::<$S>();
        let parts = components(&x, &y);
        let scalar: Vec<_> = parts
            .iter()
            .map(|values| $build(x.clone(), y.clone(), values.clone()).unwrap())
            .collect();
        let arrays: Vec<[$S; 3]> = (0..parts[0].len())
            .map(|i| [parts[0][i], parts[1][i], parts[2][i]])
            .collect();
        let complex: Vec<Complex<$S>> = (0..parts[0].len())
            .map(|i| Complex::new(parts[0][i], parts[1][i]))
            .collect();
        let vectors: Vec<Vec<$S>> = arrays.iter().map(|value| value.to_vec()).collect();
        let array_model = $build(x.clone(), y.clone(), arrays).unwrap();
        let complex_model = $build(x.clone(), y.clone(), complex).unwrap();
        let vector_model = $build(x.clone(), y.clone(), vectors).unwrap();
        for point in points::<$S>() {
            let expected: Vec<$S> = scalar
                .iter()
                .map(|model| model.evaluate(point, $policy).unwrap())
                .collect();
            assert_eq!(
                array_model.evaluate(point, $policy).unwrap().to_vec(),
                expected
            );
            assert_eq!(vector_model.evaluate(point, $policy).unwrap(), expected);
            let value = complex_model.evaluate(point, $policy).unwrap();
            assert_eq!([value.re, value.im], [expected[0], expected[1]]);
        }
        (scalar, array_model, vector_model)
    }};
}

fn qualify<S: RealScalar>() {
    let clamp = ExtrapolationPolicy::Clamp;
    let _ = check_family!(S, BilinearInterpolator::new, clamp);
    let _ = check_family!(S, BicubicInterpolator::monotone, clamp);
    let (scalar, arrays, vectors) = check_family!(S, BicubicInterpolator::new, clamp);
    for point in points::<S>() {
        let gradients: Vec<[S; 2]> = scalar
            .iter()
            .map(|model| model.gradient(point, clamp).unwrap())
            .collect();
        let hessians: Vec<[[S; 2]; 2]> = scalar
            .iter()
            .map(|model| model.hessian(point, clamp).unwrap())
            .collect();
        let gradient = arrays.gradient(point, clamp).unwrap();
        let hessian = vectors.hessian(point, clamp).unwrap();
        for component in 0..3 {
            for axis in 0..2 {
                assert_eq!(gradient[axis][component], gradients[component][axis]);
                for other in 0..2 {
                    assert_eq!(
                        hessian[axis][other][component],
                        hessians[component][axis][other]
                    );
                }
            }
        }
    }
}

#[test]
fn affine_values_match_component_evaluation_in_both_precisions() {
    qualify::<f32>();
    qualify::<f64>();
}

#[test]
fn value_shapes_are_checked_before_writing() {
    let model =
        BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![vec![0.0, 1.0]; 4])
            .unwrap();
    let mut workspace = model.workspace().unwrap();
    let mut wrong = [99.0; 3];
    let error = model
        .evaluate_into(
            [0.5, 0.5],
            ExtrapolationPolicy::Reject,
            &mut wrong[..],
            &mut workspace,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(wrong, [99.0; 3]);
    assert_eq!(model.value_dimension(), 2);
    assert_eq!(model.zero_value().unwrap(), vec![0.0, 0.0]);
    let ragged = BilinearInterpolator::new(
        vec![0.0_f64, 1.0],
        vec![0.0, 1.0],
        vec![vec![0.0], vec![0.0, 1.0], vec![0.0], vec![0.0]],
    );
    assert_eq!(
        ragged.unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let complex = BicubicInterpolator::new(
        vec![0.0_f64, 1.0],
        vec![0.0, 1.0],
        vec![Complex::new(1.0, f64::NAN); 4],
    );
    assert_eq!(complex.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
    assert_eq!(Complex::new(0.0_f64, 0.0).dimension().unwrap(), 2);
}
