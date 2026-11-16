//! Exact extrema and decorations across square and quotient sign branches.
#![cfg(test)]

use scioxide_core::RealScalar;
use scioxide_validated::{DecoratedInterval as D, Decoration};

fn assert_bounds<S: RealScalar>(result: D<S>, lo: S, hi: S, decoration: Decoration) {
    let (lower, upper) = result.set().bounds().unwrap();
    assert_eq!(
        lower.unwrap_or(-S::INFINITY).into_f64().to_bits(),
        lo.into_f64().to_bits()
    );
    assert_eq!(
        upper.unwrap_or(S::INFINITY).into_f64().to_bits(),
        hi.into_f64().to_bits()
    );
    assert_eq!(result.decoration(), decoration);
}

fn extrema<S: RealScalar>() {
    let zero = S::ZERO;
    let one = S::ONE;
    let two = S::TWO;
    let entire = D::<S>::entire();
    let empty = D::<S>::empty();
    let positive = D::from(scioxide_core::ExtendedInterval::new(Some(one), None).unwrap());
    let negative = D::from(scioxide_core::ExtendedInterval::new(None, Some(-one)).unwrap());

    {
        for (lo, hi, squared_lo, squared_hi) in [
            (1., 2., 1., 4.),
            (-2., -1., 1., 4.),
            (-2., 1., 0., 4.),
            (-1., 2., 0., 4.),
            (-2., 2., 0., 4.),
            (-2., 0., 0., 4.),
            (0., 2., 0., 4.),
            (-0., 0., 0., 0.),
        ] {
            let expected_lo = S::try_from_f64_exact(squared_lo).unwrap();
            let expected_hi = S::try_from_f64_exact(squared_hi).unwrap();
            for decoration in [
                Decoration::Trv,
                Decoration::Def,
                Decoration::Dac,
                Decoration::Com,
            ] {
                let x = D::with_decoration(
                    D::<S>::new(
                        S::try_from_f64_exact(lo).unwrap(),
                        S::try_from_f64_exact(hi).unwrap(),
                    )
                    .unwrap()
                    .set(),
                    decoration,
                )
                .unwrap();
                assert_bounds(
                    x.square().unwrap(),
                    if expected_lo == zero {
                        -zero
                    } else {
                        expected_lo
                    },
                    expected_hi,
                    decoration,
                );
            }
        }

        for (a, b, c, d, lo, hi) in [
            (2., 4., 1., 2., 1., 4.),
            (-4., -2., 1., 2., -4., -1.),
            (-2., 4., 1., 2., -2., 4.),
            (2., 4., -2., -1., -4., -1.),
            (-4., -2., -2., -1., 1., 4.),
            (-2., 4., -2., -1., -4., 2.),
        ] {
            for decoration in [
                Decoration::Trv,
                Decoration::Def,
                Decoration::Dac,
                Decoration::Com,
            ] {
                let numerator = D::with_decoration(
                    D::<S>::new(
                        S::try_from_f64_exact(a).unwrap(),
                        S::try_from_f64_exact(b).unwrap(),
                    )
                    .unwrap()
                    .set(),
                    decoration,
                )
                .unwrap();
                let denominator = D::with_decoration(
                    D::<S>::new(
                        S::try_from_f64_exact(c).unwrap(),
                        S::try_from_f64_exact(d).unwrap(),
                    )
                    .unwrap()
                    .set(),
                    decoration,
                )
                .unwrap();
                let lo = S::try_from_f64_exact(lo).unwrap();
                let hi = S::try_from_f64_exact(hi).unwrap();
                assert_bounds(
                    numerator
                        .div(
                            &D::new(
                                S::try_from_f64_exact(c).unwrap(),
                                S::try_from_f64_exact(d).unwrap(),
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    lo,
                    hi,
                    decoration,
                );
                assert_bounds(
                    D::new(
                        S::try_from_f64_exact(a).unwrap(),
                        S::try_from_f64_exact(b).unwrap(),
                    )
                    .unwrap()
                    .div(&denominator)
                    .unwrap(),
                    lo,
                    hi,
                    decoration,
                );
            }
        }

        let point_zero = D::<S>::point(S::ZERO).unwrap();
        for denominator in [
            D::new(-S::ONE, S::ONE).unwrap(),
            D::new(-S::ONE, S::ZERO).unwrap(),
            D::new(S::ZERO, S::ONE).unwrap(),
            D::new(S::try_from_f64_exact(-0.).unwrap(), S::ONE).unwrap(),
            entire,
        ] {
            assert_bounds(
                point_zero.div(&denominator).unwrap(),
                -zero,
                zero,
                Decoration::Trv,
            );
        }
        for numerator in [
            D::<S>::new(S::ONE, S::TWO).unwrap(),
            D::new(-S::TWO, -S::ONE).unwrap(),
            D::new(-S::ONE, S::ONE).unwrap(),
        ] {
            assert_bounds(
                numerator.div(&D::new(-S::ONE, S::ONE).unwrap()).unwrap(),
                -S::INFINITY,
                S::INFINITY,
                Decoration::Trv,
            );
            let undefined = numerator.div(&point_zero).unwrap();
            assert!(undefined.set().is_empty());
            assert_eq!(undefined.decoration(), Decoration::Trv);
        }
        assert_bounds(
            D::<S>::new(S::ONE, S::TWO)
                .unwrap()
                .div(&D::new(S::try_from_f64_exact(-0.).unwrap(), S::ONE).unwrap())
                .unwrap(),
            one,
            S::INFINITY,
            Decoration::Trv,
        );
        assert_bounds(
            D::<S>::new(S::ONE, S::TWO)
                .unwrap()
                .div(&D::new(S::ZERO, S::ONE).unwrap())
                .unwrap(),
            one,
            S::INFINITY,
            Decoration::Trv,
        );
        assert_bounds(
            D::<S>::new(S::ONE, S::TWO)
                .unwrap()
                .div(&D::new(-S::ONE, S::ZERO).unwrap())
                .unwrap(),
            -S::INFINITY,
            -one,
            Decoration::Trv,
        );
        assert_bounds(
            positive.div(&positive).unwrap(),
            -zero,
            S::INFINITY,
            Decoration::Dac,
        );
        assert_bounds(
            negative.div(&positive).unwrap(),
            -S::INFINITY,
            zero,
            Decoration::Dac,
        );
        assert!(empty.square().unwrap().set().is_empty());
        for x in [entire, positive, negative, point_zero, empty] {
            assert!(x.div(&empty).unwrap().set().is_empty());
            assert!(empty.div(&x).unwrap().set().is_empty());
        }
        assert_bounds(
            entire.square().unwrap(),
            -zero,
            S::INFINITY,
            Decoration::Dac,
        );
        for x in [positive, negative] {
            assert_bounds(x.square().unwrap(), one, S::INFINITY, Decoration::Dac);
        }
        let tiny = D::point(S::MIN_SUBNORMAL).unwrap();
        let maximum = D::point(S::MAX).unwrap();
        assert_bounds(
            tiny.square().unwrap(),
            -zero,
            S::MIN_SUBNORMAL,
            Decoration::Com,
        );
        assert_bounds(
            maximum.square().unwrap(),
            S::MAX,
            S::INFINITY,
            Decoration::Dac,
        );
        assert_bounds(
            tiny.div(&D::point(two).unwrap()).unwrap(),
            -zero,
            S::MIN_SUBNORMAL,
            Decoration::Com,
        );
        assert_bounds(
            maximum
                .div(&D::point(S::try_from_f64_exact(0.5).unwrap()).unwrap())
                .unwrap(),
            S::MAX,
            S::INFINITY,
            Decoration::Dac,
        );
    };
}

#[test]
fn extrema_f32() {
    extrema::<f32>();
}

#[test]
fn extrema_f64() {
    extrema::<f64>();
}
