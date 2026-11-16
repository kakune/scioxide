//! Independent finite-cell oracle for the public real interval set algebra.
//! Every boundary and every gap, including both tails, has a witness.

use scioxide_core::{ExtendedInterval, Interval, RealScalar, SciOxideError};

fn set_laws<S: RealScalar>() -> Result<(), SciOxideError> {
    let zero = S::ZERO;
    let one = S::ONE;
    let two = S::TWO;
    let half = one / two;
    let endpoints = [-one, zero, one];
    let witnesses = [-two, -one, -half, zero, half, one, two];
    // Membership is computed independently from integer boundary positions.
    let mut cases = vec![(ExtendedInterval::<S>::empty(), [false; 7])];
    for lower in 0_usize..=3 {
        for upper in 0_usize..=3 {
            if lower > 0 && upper < lower - 1 {
                continue;
            }
            let lo = lower.checked_sub(1).map(|i| endpoints[i]);
            let hi = endpoints.get(upper).copied();
            let set = ExtendedInterval::new(lo, hi)?;
            let membership = core::array::from_fn(|i| {
                (lower == 0 || i >= 2 * lower - 1) && (upper == 3 || i <= 2 * upper + 1)
            });
            cases.push((set, membership));
        }
    }
    for (a, am) in &cases {
        for (x, expected) in witnesses.iter().zip(am) {
            assert_eq!(a.contains(*x), *expected);
        }
        for (b, bm) in &cases {
            let intersection = a.intersection(b);
            let hull = a.hull(b);
            let union: [bool; 7] = core::array::from_fn(|i| am[i] || bm[i]);
            let first = union.iter().position(|x| *x);
            let last = union.iter().rposition(|x| *x);
            for (i, x) in witnesses.iter().enumerate() {
                assert_eq!(intersection.contains(*x), am[i] && bm[i]);
                assert_eq!(
                    hull.contains(*x),
                    first.zip(last).is_some_and(|(lo, hi)| lo <= i && i <= hi)
                );
            }
            assert_eq!(
                a.contains_interval(b),
                am.iter().zip(bm).all(|(a, b)| !b || *a)
            );
            assert_eq!(intersection, b.intersection(a));
            assert_eq!(hull, b.hull(a));
            assert_eq!(a.intersection(&hull), *a);
            assert_eq!(a.hull(&intersection), *a);
            if let (Ok(af), Ok(bf)) = (Interval::try_from(*a), Interval::try_from(*b)) {
                assert_eq!(ExtendedInterval::from(af.hull(&bf)), hull);
                assert_eq!(af.intersection(&bf), intersection);
                assert_eq!(af.contains_interval(&bf), a.contains_interval(b));
            }
            for (c, _) in &cases {
                assert_eq!(a.hull(&b.hull(c)), a.hull(b).hull(c));
                assert_eq!(
                    a.intersection(&b.intersection(c)),
                    a.intersection(b).intersection(c)
                );
            }
        }
        assert_eq!(a.hull(a), *a);
        assert_eq!(a.intersection(a), *a);
        assert_eq!(a.hull(&ExtendedInterval::empty()), *a);
        assert_eq!(a.intersection(&ExtendedInterval::entire()), *a);
        assert!(a.intersection(&ExtendedInterval::empty()).is_empty());
        assert!(a.hull(&ExtendedInterval::entire()).is_entire());
    }
    Ok(())
}

#[test]
fn independent_set_algebra_f32() -> Result<(), SciOxideError> {
    set_laws::<f32>()
}
#[test]
fn independent_set_algebra_f64() -> Result<(), SciOxideError> {
    set_laws::<f64>()
}
