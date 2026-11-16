#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_random::{
    CenteredDiscrepancy, DiscrepancyMethod as D, GeometricDiscrepancy as G, discrepancy,
    geometric_discrepancy,
};
fn check<S: RealScalar>() {
    let half = S::try_from_f64_lossy(0.5).unwrap();
    // Integrating the one-point empirical CDF gives integral (F(x)-x)^2 = 1/12.
    for (method, expected) in [
        (D::Centered, 1.0 / 12.0),
        (D::WrapAround, 1.0 / 6.0),
        (D::Mixture, 1.0 / 8.0),
        (D::L2Star, (1.0_f64 / 12.0).sqrt()),
    ] {
        assert!((discrepancy(&[half], 1, method).unwrap() - expected).abs() < 1e-14);
    }
    let points: Vec<S> = [0.25, 0.25, 0.75, 0.75]
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let mut state = CenteredDiscrepancy::new(&points[..2], 2, 3).unwrap();
    let expected = discrepancy(&points, 2, D::Centered).unwrap();
    assert!((state.evaluate_append(&points[2..]).unwrap() - expected).abs() < 1e-14);
    assert_eq!(state.points(), &points[..2]);
    assert!((state.append(&points[2..]).unwrap() - expected).abs() < 1e-14);
    assert_eq!(state.points(), points);
    let before = state.points().to_vec();
    assert!(state.append(&[S::ONE]).is_err());
    assert_eq!(state.points(), before);
    let square: Vec<S> = [0., 0., 0., 1., 1., 0., 1., 1.]
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    assert_eq!(
        geometric_discrepancy(&square, 2, G::MinimumDistance).unwrap(),
        1.0
    );
    assert_eq!(
        geometric_discrepancy(&square, 2, G::MeanSpanningTreeEdge).unwrap(),
        1.0
    );
    assert_eq!(
        geometric_discrepancy(&[half, half], 1, G::MeanSpanningTreeEdge).unwrap(),
        0.0
    );
    // Five points on a line: minimum gap 0.1, spanning tree edges 0.1, 0.2, 0.3, 0.4.
    let line: Vec<S> = [0.0, 0.1, 0.3, 0.6, 1.0]
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let tolerance = 4.0 * S::EPSILON.into_f64();
    assert!((geometric_discrepancy(&line, 1, G::MinimumDistance).unwrap() - 0.1).abs() < tolerance);
    assert!(
        (geometric_discrepancy(&line, 1, G::MeanSpanningTreeEdge).unwrap() - 0.25).abs()
            < tolerance
    );
}
#[test]
fn exact_both_widths() {
    check::<f32>();
    check::<f64>();
}
#[test]
fn invalid_inputs() {
    for points in [
        vec![],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![-0.1],
        vec![1.1],
    ] {
        assert!(discrepancy(&points, 1, D::Centered).is_err());
    }
    assert!(discrepancy(&[0.5], 0, D::Centered).is_err());
    assert!(geometric_discrepancy(&[0.5], 1, G::MinimumDistance).is_err());
    assert!(geometric_discrepancy(&[0.5], 1, G::MeanSpanningTreeEdge).is_err());
    assert!(geometric_discrepancy(&[0., 1.], 0, G::MeanSpanningTreeEdge).is_err());
    let mut state = CenteredDiscrepancy::new(&[0.5], 1, 1).unwrap();
    assert!(state.append(&[0.25]).is_err());
    assert_eq!(state.points(), &[0.5]);
}
#[test]
fn analytic_stratified_cdf_integral() {
    // Exact L2-star integral for n centered equally spaced 1D points is 1/(12n²).
    for n in [2, 3, 7, 32] {
        let points: Vec<_> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let expected = 1.0 / (12.0_f64.sqrt() * n as f64);
        assert!((discrepancy(&points, 1, D::L2Star).unwrap() - expected).abs() < 2e-13);
        assert!(
            (discrepancy(&points, 1, D::Centered).unwrap() - expected * expected).abs() < 1e-14
        );
    }
}

#[test]
fn independent_rational_multidimensional_references() {
    for line in include_str!("../data/design_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let dim = values[0] as usize;
        let n = values[1] as usize;
        let points = &values[2..2 + n * dim];
        let narrow: Vec<f32> = points.iter().map(|&x| x as f32).collect();
        for (index, method) in [D::Centered, D::WrapAround, D::Mixture, D::L2Star]
            .iter()
            .enumerate()
        {
            let raw = values[2 + n * dim + index];
            let expected = if *method == D::L2Star {
                raw.sqrt()
            } else {
                raw
            };
            assert!((discrepancy(points, dim, *method).unwrap() - expected).abs() < 2e-14);
            assert!((discrepancy(&narrow, dim, *method).unwrap() - expected).abs() < 2e-14);
        }
    }
}
