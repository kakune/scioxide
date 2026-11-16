//! Exact rational Hermite values at binary64 coordinates.
#![allow(clippy::unwrap_used)]

use scioxide_poly::OrthogonalFamily;

/// Scalar H and He evaluation is as accurate as the forward three-term
/// recurrence: within `4 n ε` of the largest member magnitude `max_(k<=n)|P_k(x)|`
/// on `|x| <= 16`. The removed parity elimination in `x²` fails this bound.
#[test]
fn scalar_hermite_matches_exact_values_within_the_recurrence_bound() {
    let mut rows = 0;
    for line in include_str!("data/hermite.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let family = if fields[0] == "H" {
            OrthogonalFamily::<f64>::HermiteH
        } else {
            assert_eq!(fields[0], "He");
            OrthogonalFamily::<f64>::HermiteHe
        };
        let degree: usize = fields[1].parse().unwrap();
        let point: f64 = fields[2].parse().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        let envelope: f64 = fields[4].parse().unwrap();
        let actual = family.evaluate(degree, point).unwrap();
        let tolerance = 4.0 * degree as f64 * f64::EPSILON * envelope;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{}, n={degree}, x={point}: {actual} != {expected}, tolerance={tolerance}",
            fields[0]
        );
        rows += 1;
    }
    assert_eq!(rows, 2 * 4 * 1401);
}
