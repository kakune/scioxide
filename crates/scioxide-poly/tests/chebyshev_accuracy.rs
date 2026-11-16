//! Independent high-precision values at exact coordinates in both scalar widths.
#![allow(clippy::unwrap_used)]

use scioxide_poly::OrthogonalFamily;

#[test]
fn scalar_chebyshev_matches_high_precision_on_the_interior_grid() {
    let mut rows = 0;
    for line in include_str!("data/chebyshev.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let width: u32 = fields[0].parse().unwrap();
        let degree: usize = fields[1].parse().unwrap();
        let point: f64 = fields[2].parse().unwrap();
        for (second, column) in [(false, 3), (true, 4)] {
            let expected: f64 = fields[column].parse().unwrap();
            let (actual, epsilon) = if width == 32 {
                let family = if second {
                    OrthogonalFamily::<f32>::ChebyshevU
                } else {
                    OrthogonalFamily::<f32>::ChebyshevT
                };
                (
                    f64::from(family.evaluate(degree, point as f32).unwrap()),
                    f64::from(f32::EPSILON),
                )
            } else {
                let family = if second {
                    OrthogonalFamily::<f64>::ChebyshevU
                } else {
                    OrthogonalFamily::<f64>::ChebyshevT
                };
                (family.evaluate(degree, point).unwrap(), f64::EPSILON)
            };
            let envelope = if second { (degree + 1) as f64 } else { 1.0 };
            let tolerance = 2.0 * degree as f64 * epsilon * envelope;
            assert!(
                (actual - expected).abs() <= tolerance,
                "width={width}, n={degree}, x={point}, U={second}: {actual} != {expected}, tolerance={tolerance}"
            );
        }
        rows += 1;
    }
    assert_eq!(rows, 2 * 4 * 2001);
}
