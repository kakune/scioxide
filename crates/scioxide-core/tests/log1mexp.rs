//! Cancellation, endpoint and error contracts of logarithmic complementation.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, SciOxideErrorKind, log1mexp};

fn check<S: RealScalar>() {
    let n = |x| S::try_from_f64_lossy(x).unwrap();
    assert_eq!(log1mexp(S::ZERO).unwrap(), S::NEG_INFINITY);
    assert_eq!(log1mexp(-S::ZERO).unwrap(), S::NEG_INFINITY);
    let limit = log1mexp(S::NEG_INFINITY).unwrap();
    assert_eq!(limit, S::ZERO);
    assert!(limit.is_sign_negative());
    for x in [S::NAN, S::INFINITY] {
        assert_eq!(
            log1mexp(x).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert_eq!(
        log1mexp(S::ONE).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
    let center = -S::LN_2;
    for x in [center - S::EPSILON, center, center + S::EPSILON] {
        let y = log1mexp(x).unwrap();
        assert!((y - center).abs() <= n(8.0) * S::EPSILON);
    }
    for (input, reference) in [
        (-1.0, -0.458_675_145_387_081_9),
        (-100.0, -3.720_075_976_020_836e-44),
    ] {
        let value = log1mexp(n(input)).unwrap();
        assert!(
            (value - n(reference)).abs()
                <= n(4.0) * S::EPSILON * n(reference).abs() + S::MIN_SUBNORMAL
        );
    }
    let tiny = -S::MIN_SUBNORMAL;
    assert!(log1mexp(tiny).unwrap().is_finite());
    assert!(
        (log1mexp(tiny).unwrap() - (-tiny).ln()).abs() <= n(4.0) * S::EPSILON * (-tiny).ln().abs()
    );
}

#[test]
fn both_precisions_preserve_logarithmic_limits_and_small_complements() {
    check::<f64>();
    check::<f32>();
}
