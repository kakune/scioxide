//! Public predicate conventions, full-range inputs, and validation.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_spatial::{PredicateSign::*, incircle, insphere, orient2d, orient3d};

fn cases<S: RealScalar>() {
    let s = |x| S::try_from_f64_exact(x).unwrap();
    let p2 = |p: [f64; 2]| p.map(s);
    let p3 = |p: [f64; 3]| p.map(s);
    let [a, b, c, d] = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]].map(p2);
    let zero = p2([0., 0.]);
    assert_eq!(orient2d(&a, &b, &c).unwrap(), Positive);
    assert_eq!(orient2d(&b, &a, &c).unwrap(), Negative);
    assert_eq!(incircle(&a, &b, &c, &zero).unwrap(), Positive);
    assert_eq!(incircle(&a, &b, &c, &d).unwrap(), Zero);
    assert_eq!(incircle(&a, &b, &c, &p2([0., -2.])).unwrap(), Negative);
    assert_eq!(orient2d(&a, &a, &b).unwrap(), Zero);
    assert_eq!(orient2d(&zero, &p2([-0., 0.]), &a).unwrap(), Zero);
    let [a, b, c, d] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]].map(p3);
    assert_eq!(orient3d(&a, &b, &c, &d).unwrap(), Positive);
    assert_eq!(
        insphere(&a, &b, &c, &d, &p3([0., 0., 0.])).unwrap(),
        Positive
    );
    assert_eq!(insphere(&a, &b, &c, &d, &p3([0., -1., 0.])).unwrap(), Zero);
    assert_eq!(
        insphere(&a, &b, &c, &d, &p3([0., -2., 0.])).unwrap(),
        Negative
    );
    for scale in [S::MAX, S::MIN_POSITIVE * S::EPSILON] {
        let p = |p: [f64; 3]| p.map(|x| s(x) * scale);
        let [a, b, c, d] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]].map(p);
        assert_eq!(orient3d(&a, &b, &c, &d).unwrap(), Positive);
        assert_eq!(
            insphere(&a, &b, &c, &d, &p([0., 0., 0.])).unwrap(),
            Positive
        );
        assert_eq!(insphere(&a, &b, &c, &d, &p([0., -1., 0.])).unwrap(), Zero);
        let [a, b, c] = [[scale, s(0.)], [s(0.), scale], [-scale, s(0.)]];
        assert_eq!(orient2d(&a, &b, &c).unwrap(), Positive);
        assert_eq!(incircle(&a, &b, &c, &zero).unwrap(), Positive);
    }
    // Every coordinate position is validated, even when early rows repeat.
    for bad in [S::NAN, S::INFINITY, -S::INFINITY] {
        for position in 0..15 {
            let mut points = [p3([0., 0., 0.]); 5];
            points[position / 3][position % 3] = bad;
            assert!(
                matches ! (& (insphere (& points [0] , & points [1] , & points [2] , & points [3] , & points [4])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
            );
        }
        let bad2 = [bad, s(0.)];
        let bad3 = [bad, s(0.), s(0.)];
        assert!(
            matches ! (& (orient2d (& zero , & zero , & bad2)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
        assert!(
            matches ! (& (incircle (& zero , & zero , & zero , & bad2)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
        assert!(
            matches ! (& (orient3d (& a , & a , & a , & bad3)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
    }
}

#[test]
fn f32_contract() {
    cases::<f32>();
}
#[test]
fn f64_contract() {
    cases::<f64>();
}

fn every_nonfinite_position<S: RealScalar>() {
    for bad in [S::NAN, S::INFINITY, -S::INFINITY] {
        for position in 0..15 {
            // Earlier out-of-envelope coordinates must not skip later validation.
            let mut p = [[S::MAX; 3]; 5];
            p[position / 3][position % 3] = bad;
            let q = p.map(|p| [p[0], p[1]]);
            let mut errors = vec![insphere(&p[0], &p[1], &p[2], &p[3], &p[4]).unwrap_err()];
            if position < 12 {
                errors.push(orient3d(&p[0], &p[1], &p[2], &p[3]).unwrap_err());
            }
            if position % 3 < 2 && position < 12 {
                errors.push(incircle(&q[0], &q[1], &q[2], &q[3]).unwrap_err());
                if position < 9 {
                    errors.push(orient2d(&q[0], &q[1], &q[2]).unwrap_err());
                }
            }
            for error in errors {
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: scioxide_core::NonFiniteOrigin::Input,
                        ..
                    }
                ));
            }
        }
    }
}

#[test]
fn every_argument_position_rejects_nonfinite_input() {
    every_nonfinite_position::<f32>();
    every_nonfinite_position::<f64>();
}

fn simplex_workspace<S: RealScalar>() {
    use scioxide_core::{Budget, SciOxideErrorKind};
    use scioxide_spatial::SimplexPredicateWorkspace;
    assert!(SimplexPredicateWorkspace::new(0, Budget::ONE).is_err());
    assert!(SimplexPredicateWorkspace::new(usize::MAX, Budget::ONE).is_err());
    for dimension in 1..=8 {
        let terms = (1..=dimension + 1).product::<usize>();
        assert!(
            SimplexPredicateWorkspace::new(dimension, Budget::try_from(terms - 1).unwrap())
                .is_err()
        );
        let mut workspace =
            SimplexPredicateWorkspace::new(dimension, Budget::try_from(terms).unwrap()).unwrap();
        let mut points = vec![vec![S::ZERO; dimension]; dimension + 1];
        for axis in 0..dimension {
            points[axis + 1][axis] = S::ONE;
        }
        let orient = |workspace: &mut SimplexPredicateWorkspace, points: &[Vec<S>]| {
            workspace.orient(&points.iter().map(Vec::as_slice).collect::<Vec<_>>())
        };
        for _ in 0..3 {
            assert_eq!(orient(&mut workspace, &points).unwrap(), Positive);
        }
        points.swap(0, 1);
        assert_eq!(orient(&mut workspace, &points).unwrap(), Negative);
        points[0] = points[1].clone();
        for _ in 0..3 {
            assert_eq!(orient(&mut workspace, &points).unwrap(), Zero);
        }
        assert!(workspace.orient(&[&[S::ZERO]]).is_err());
        let saved = points[dimension][dimension - 1];
        for bad in [S::NAN, S::INFINITY, -S::INFINITY] {
            points[dimension][dimension - 1] = bad;
            assert_eq!(
                orient(&mut workspace, &points).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
        }
        points[dimension][dimension - 1] = saved;
        assert_eq!(orient(&mut workspace, &points).unwrap(), Zero);
    }
    let mut workspace = SimplexPredicateWorkspace::new(2, Budget::try_from(6).unwrap()).unwrap();
    let p = [[S::ZERO; 2], [S::ONE, S::ZERO], [S::ZERO, S::ONE]];
    assert_eq!(
        workspace.orient(&[&p[0], &p[1], &p[2]]).unwrap(),
        orient2d(&p[0], &p[1], &p[2]).unwrap()
    );
    let mut workspace = SimplexPredicateWorkspace::new(3, Budget::try_from(24).unwrap()).unwrap();
    let p = [
        [S::ZERO; 3],
        [S::ONE, S::ZERO, S::ZERO],
        [S::ZERO, S::ONE, S::ZERO],
        [S::ZERO, S::ZERO, S::ONE],
    ];
    assert_eq!(
        workspace.orient(&[&p[0], &p[1], &p[2], &p[3]]).unwrap(),
        Positive
    );
    assert_eq!(orient3d(&p[0], &p[1], &p[2], &p[3]).unwrap(), Negative);
}
#[test]
fn simplex_workspace_constructor_admission_and_conventions() {
    simplex_workspace::<f32>();
    simplex_workspace::<f64>();
}
