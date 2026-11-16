//! Public near-integer semantics for the real-order Bessel family.

#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_core::SciOxideError;
use scioxide_special::{
    bessel_i, bessel_i_scaled, bessel_j, bessel_k, bessel_k_scaled, bessel_y, hankel_1, hankel_2,
};

fn assert_domain<T>(result: Result<T, SciOxideError>) {
    {
        let outcome = &(result);
        let Err(err) = outcome else {
            panic!("expected error");
        };
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

macro_rules! exercise_near_integer_semantics {
    ($scalar:ty) => {{
        let centers = [-1.0 as $scalar, 0.0 as $scalar, 1.0 as $scalar];
        let small = 0.125 as $scalar;
        let negative = -1.0 as $scalar;
        let asymptotic = 50.0 as $scalar;

        centers.into_iter().for_each(|center| {
            let neighbors = [center.next_down(), center.next_up()];

            assert!(bessel_j(center, negative).unwrap().is_finite());
            assert!(bessel_i(center, negative).unwrap().is_finite());
            assert!(bessel_i_scaled(center, negative).unwrap().is_finite());
            neighbors.into_iter().for_each(|order| {
                assert_domain(bessel_j(order, negative));
                assert_domain(bessel_i(order, negative));
                assert_domain(bessel_i_scaled(order, negative));
            });

            [center.next_down(), center, center.next_up()]
                .into_iter()
                .for_each(|order| {
                    assert_domain(bessel_y(order, negative));
                    assert_domain(hankel_1(order, negative));
                    assert_domain(hankel_2(order, negative));
                    assert_domain(bessel_k(order, negative));
                    assert_domain(bessel_k_scaled(order, negative));

                    assert_domain(bessel_y(order, 0.0 as $scalar));
                    assert_domain(hankel_1(order, 0.0 as $scalar));
                    assert_domain(hankel_2(order, 0.0 as $scalar));
                    assert_eq!(
                        bessel_k(order, 0.0 as $scalar).unwrap(),
                        <$scalar>::INFINITY
                    );
                    assert_eq!(
                        bessel_k_scaled(order, 0.0 as $scalar).unwrap(),
                        <$scalar>::INFINITY
                    );
                });

            assert!(bessel_y(center, small).unwrap().is_finite());
            assert!(bessel_k(center, small).unwrap().is_finite());
            assert!(bessel_k_scaled(center, small).unwrap().is_finite());
            assert!(hankel_1(center, small).unwrap().re.is_finite());
            assert!(hankel_2(center, small).unwrap().re.is_finite());
            neighbors.into_iter().for_each(|order| {
                assert!(bessel_y(order, small).unwrap().is_finite());
                assert!(bessel_k(order, small).unwrap().is_finite());
                assert!(bessel_k_scaled(order, small).unwrap().is_finite());
                assert!(hankel_1(order, small).unwrap().im.is_finite());
                assert!(hankel_2(order, small).unwrap().im.is_finite());
            });

            [center.next_down(), center, center.next_up()]
                .into_iter()
                .for_each(|order| {
                    assert!(bessel_j(order, small).unwrap().is_finite());
                    assert!(bessel_i(order, small).unwrap().is_finite());
                    assert!(bessel_i_scaled(order, small).unwrap().is_finite());

                    let j = bessel_j(order, asymptotic).unwrap();
                    let y = bessel_y(order, asymptotic).unwrap();
                    let i = bessel_i(order, asymptotic).unwrap();
                    let i_scaled = bessel_i_scaled(order, asymptotic).unwrap();
                    let k = bessel_k(order, asymptotic).unwrap();
                    let k_scaled = bessel_k_scaled(order, asymptotic).unwrap();
                    let h1 = hankel_1(order, asymptotic).unwrap();
                    let h2 = hankel_2(order, asymptotic).unwrap();

                    assert!(
                        [j, y, i, i_scaled, k, k_scaled]
                            .into_iter()
                            .all(<$scalar>::is_finite)
                    );
                    assert_eq!(h1.re, j);
                    assert_eq!(h1.im, y);
                    assert_eq!(h2.re, j);
                    assert_eq!(h2.im, -y);
                });
        });

        assert_domain(bessel_j((-1.0 as $scalar).next_down(), 0.0 as $scalar));
        assert_domain(bessel_j((-1.0 as $scalar).next_up(), 0.0 as $scalar));
        assert_eq!(bessel_j(-1.0 as $scalar, 0.0 as $scalar).unwrap(), 0.0);
        assert_domain(bessel_i((-1.0 as $scalar).next_down(), 0.0 as $scalar));
        assert_domain(bessel_i((-1.0 as $scalar).next_up(), 0.0 as $scalar));
        assert_eq!(bessel_i(-1.0 as $scalar, 0.0 as $scalar).unwrap(), 0.0);
        assert_domain(bessel_i_scaled(
            (-1.0 as $scalar).next_down(),
            0.0 as $scalar,
        ));
        assert_domain(bessel_i_scaled((-1.0 as $scalar).next_up(), 0.0 as $scalar));
        assert_eq!(
            bessel_i_scaled(-1.0 as $scalar, 0.0 as $scalar).unwrap(),
            0.0
        );

        assert_domain(bessel_j((0.0 as $scalar).next_down(), 0.0 as $scalar));
        assert_eq!(bessel_j(0.0 as $scalar, 0.0 as $scalar).unwrap(), 1.0);
        assert_eq!(
            bessel_j((0.0 as $scalar).next_up(), 0.0 as $scalar).unwrap(),
            0.0
        );
        assert_domain(bessel_i((0.0 as $scalar).next_down(), 0.0 as $scalar));
        assert_eq!(bessel_i(0.0 as $scalar, 0.0 as $scalar).unwrap(), 1.0);
        assert_eq!(
            bessel_i((0.0 as $scalar).next_up(), 0.0 as $scalar).unwrap(),
            0.0
        );
        assert_domain(bessel_i_scaled(
            (0.0 as $scalar).next_down(),
            0.0 as $scalar,
        ));
        assert_eq!(
            bessel_i_scaled(0.0 as $scalar, 0.0 as $scalar).unwrap(),
            1.0
        );
        assert_eq!(
            bessel_i_scaled((0.0 as $scalar).next_up(), 0.0 as $scalar).unwrap(),
            0.0
        );

        [
            (1.0 as $scalar).next_down(),
            1.0 as $scalar,
            (1.0 as $scalar).next_up(),
        ]
        .into_iter()
        .for_each(|order| {
            assert_eq!(bessel_j(order, 0.0 as $scalar).unwrap(), 0.0);
            assert_eq!(bessel_i(order, 0.0 as $scalar).unwrap(), 0.0);
            assert_eq!(bessel_i_scaled(order, 0.0 as $scalar).unwrap(), 0.0);
        });
    }};
}

#[test]
fn f64_orders_adjacent_to_integers_keep_their_own_semantics() {
    exercise_near_integer_semantics!(f64);
}

#[test]
fn f32_orders_adjacent_to_integers_keep_their_own_semantics() {
    exercise_near_integer_semantics!(f32);
}

#[test]
fn large_orders_classify_range_before_recurrence() {
    let order = 2.0_f64.powi(63);
    assert!(bessel_j(order, order).unwrap() > 0.0);
    let error = bessel_y(order, 1.0).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    {
        let outcome = &(bessel_k(order, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }
    assert_eq!(bessel_i(order, -1.0).unwrap(), 0.0);
    let order = 2.0_f32.powi(63);
    assert!(bessel_j(order, order).unwrap() > 0.0);
    let error = bessel_y(order, 1.0).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    {
        let outcome = &(bessel_k(order, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }
    assert_eq!(bessel_i(order, -1.0).unwrap(), 0.0);
}
