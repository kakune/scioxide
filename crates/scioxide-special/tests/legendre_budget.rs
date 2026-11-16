//! Large degrees are bounded by work allowances rather than a second cap.
#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar, SciOxideErrorView};
use scioxide_special::{
    LegendreOptions, legendre_p, legendre_p_assoc, legendre_p_with_options, legendre_q,
    legendre_q_with_options,
};

#[test]
fn degree_twenty_thousand_matches_independent_references() {
    fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S) {
        let ample = LegendreOptions::default()
            .with_iteration_budget(Budget::try_from(100_000_000).unwrap());
        for row in include_str!("data/legendre_large_degree.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = row.split_whitespace().collect();
            if fields[0] != scalar {
                continue;
            }
            let degree = fields[1].parse().unwrap();
            let x = decode(fields[2]);
            let p = legendre_p_with_options(degree, x, ample).unwrap();
            let q = legendre_q_with_options(degree, x, ample).unwrap();
            for (report, expected) in [(&p, decode(fields[3])), (&q, decode(fields[4]))] {
                let actual = report.try_value().unwrap();
                let spacing = expected.abs().next_after(S::INFINITY) - expected.abs();
                assert!(
                    (actual - expected).abs() <= S::try_from_f64_lossy(64.0).unwrap() * spacing,
                    "{row}: {actual:?}"
                );
                assert!(report.counts().iterations() >= degree - 1);
            }
            let actual = p.try_value().unwrap();
            let associated = legendre_p_assoc(degree, 0, x).unwrap();
            let spacing = actual.abs().next_after(S::INFINITY) - actual.abs();
            assert!((actual - associated).abs() <= S::try_from_f64_lossy(64.0).unwrap() * spacing);
            assert_eq!(actual, legendre_p(degree, x).unwrap());
            assert_eq!(q.try_value().unwrap(), legendre_q(degree, x).unwrap());
            for limit in [1, 19998] {
                let options = ample.with_iteration_budget(Budget::try_from(limit).unwrap());
                for error in [
                    legendre_p_with_options(degree, x, options).unwrap_err(),
                    legendre_q_with_options(degree, x, options).unwrap_err(),
                ] {
                    assert_eq!(publication::exhausted_counts(&error).iterations(), limit);
                }
            }
        }
    }
    check("f64", |bits| {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    });
    check("f32", |bits| {
        f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
    });
}

#[test]
fn representation_boundaries_are_checked_without_integer_overflow() {
    assert_eq!(legendre_p(usize::MAX, -1.0_f64).unwrap(), -1.0);
    if usize::BITS > 52 {
        for error in [
            legendre_p(usize::MAX, 0.3_f64).unwrap_err(),
            legendre_q(usize::MAX, 0.3_f64).unwrap_err(),
        ] {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput { .. }
            ));
        }
    }
}

#[test]
fn associated_orders_charge_recurrence_and_negative_order_reduction() {
    for (order, steps) in [(3, 10), (-3, 16), (-10, 30)] {
        let result = scioxide_special::legendre_p_assoc_with_options(
            10,
            order,
            0.3_f64,
            LegendreOptions::default(),
        )
        .unwrap();
        assert_eq!(result.counts().iterations(), steps);
    }
}
