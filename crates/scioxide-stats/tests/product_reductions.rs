//! Centered product blocks, exact cancellation, and native/extended agreement.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::Matrix;
use scioxide_stats::*;

fn close<S: RealScalar>(actual: S, expected: S) {
    let tolerance = 256.0 * S::EPSILON.into_f64() * expected.into_f64().abs().max(1.0);
    assert!(
        (actual.into_f64() - expected.into_f64()).abs() <= tolerance,
        "{actual:?} != {expected:?}"
    );
}

fn paired<S: RealScalar>() {
    for n in [0_usize, 1, 2, 3, 4, 5, 255, 256, 257, 511, 512, 513] {
        let x: Vec<_> = (0..n)
            .map(|i| S::try_from_f64_exact(1_048_576.0 + (i % 17) as f64 * 0.25).unwrap())
            .collect();
        let y: Vec<_> = (0..n)
            .map(|i| S::try_from_f64_exact(-2_097_152.0 + (i % 11) as f64 * 0.5).unwrap())
            .collect();
        for order in [MomentOrder::Second, MomentOrder::Third, MomentOrder::Fourth] {
            let batch = OnlineCovariance::from_slices(&x, &y, order)
                .unwrap()
                .snapshot();
            let mut stream = OnlineCovariance::new(order);
            for (&x, &y) in x.iter().zip(&y) {
                stream.observe(x, y).unwrap();
            }
            let stream = stream.snapshot();
            assert_eq!(batch.count(), stream.count());
            for (actual, expected) in [
                (batch.mean_x().unwrap(), stream.mean_x().unwrap()),
                (batch.mean_y().unwrap(), stream.mean_y().unwrap()),
                (
                    batch.sample_covariance().unwrap(),
                    stream.sample_covariance().unwrap(),
                ),
                (batch.correlation().unwrap(), stream.correlation().unwrap()),
                (
                    batch.marginal_x().population_variance().unwrap(),
                    stream.marginal_x().population_variance().unwrap(),
                ),
            ] {
                assert_eq!(actual.is_some(), expected.is_some());
                if let (Some(actual), Some(expected)) = (actual, expected) {
                    close(actual, expected);
                }
            }
        }
        assert_eq!(
            OnlineCovariance::from_slices(&x, &y[..n.saturating_sub(1)], MomentOrder::Second)
                .is_ok(),
            n == 0
        );
    }
}

#[test]
fn paired_blocks_match_streaming_in_both_widths() {
    paired::<f32>();
    paired::<f64>();
}

#[test]
fn shift_residuals_and_exact_line_fits_are_retained() {
    let state = OnlineCovariance::from_slices(
        &[1e16, 1.0, -1e16],
        &[1e16, 2.0, -1e16],
        MomentOrder::Fourth,
    )
    .unwrap()
    .snapshot();
    assert!((state.mean_x().unwrap().unwrap() - 1.0_f64 / 3.0).abs() < 1e-16);
    assert!((state.mean_y().unwrap().unwrap() - 2.0_f64 / 3.0).abs() < 1e-16);
    let x = core::array::from_fn::<_, 257, _>(|i| 1_048_576.0 + i as f64 * 0.25);
    let y = x.map(|x| 3.0 * x + 7.0);
    let line = linear_regression(&x, &y).unwrap().line();
    assert!((line.slope - 3.0).abs() < 1e-14);
    assert!((line.intercept - 7.0).abs() < 1e-14);
    let state = OnlineCovariance::from_slices(&[-0.0_f64], &[0.0], MomentOrder::Second)
        .unwrap()
        .snapshot();
    let mut stream = OnlineCovariance::new(MomentOrder::Second);
    stream.observe(-0.0_f64, 0.0).unwrap();
    assert_eq!(
        state.mean_x().unwrap().unwrap().to_bits(),
        stream.snapshot().mean_x().unwrap().unwrap().to_bits()
    );
    assert_eq!(
        state.marginal_x().minimum().unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn exact_products_survive_cancelling_cross_moments() {
    let e = 2.0_f64.powi(-27);
    let x = [1.0 + e, 1.0, -(1.0 + e), -1.0];
    let y = [1.0 - e, -1.0, -(1.0 - e), 1.0];
    let state = OnlineCovariance::from_slices(&x, &y, MomentOrder::Second).unwrap();
    let expected = -2.0 * e * e / 3.0;
    assert!(
        (state.snapshot().sample_covariance().unwrap().unwrap() / expected - 1.0).abs() < 1e-14
    );
    let records: Vec<_> = x.into_iter().zip(y).flat_map(|(x, y)| [x, y]).collect();
    let data = Matrix::from_vec(4, 2, records).unwrap();
    let covariance = sample_covariance_with_options(data.view(), MultivariateOptions::new())
        .unwrap()
        .1;
    assert!((covariance.as_slice()[1] / expected - 1.0).abs() < 1e-14);
}

#[test]
fn paired_errors_and_extended_diagnostics_are_independent() {
    let mut x = [1.0_f64; 513];
    let y = [2.0; 513];
    for position in [0, 3, 255, 256, 512] {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            x[position] = invalid;
            assert!(OnlineCovariance::from_slices(&x, &y, MomentOrder::Second).is_err());
        }
        x[position] = 1.0;
    }
    let state = OnlineCovariance::from_slices(
        &[-f64::MAX, f64::MAX],
        &[-f64::MAX, f64::MAX],
        MomentOrder::Second,
    )
    .unwrap()
    .snapshot();
    assert_eq!(state.correlation().unwrap(), Some(1.0));
    assert!(state.sample_covariance().is_err());
    let tiny = f64::from_bits(1);
    let state = OnlineCovariance::from_slices(&[-tiny, tiny], &[tiny, -tiny], MomentOrder::Second)
        .unwrap()
        .snapshot();
    assert_eq!(
        state.population_covariance().unwrap().unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(state.correlation().unwrap(), Some(-1.0));
}

fn weighted<T: NumericScalar>() {
    for n in [3, 255, 256, 257, 513] {
        let counts: Vec<_> = (0..n).map(|i| (i % 4) as u64).collect();
        let records: Vec<_> = (0..n)
            .flat_map(|i| [i % 7, i % 11])
            .map(|i| {
                T::from_components(
                    T::Real::try_from_f64_exact(i as f64 * 0.25).unwrap(),
                    T::Real::try_from_f64_exact(i as f64 * 0.125).unwrap(),
                )
            })
            .collect();
        let repeated: Vec<_> = records
            .chunks_exact(2)
            .zip(&counts)
            .flat_map(|(row, &count)| (0..count).flat_map(move |_| row.iter().copied()))
            .collect();
        let data = Matrix::from_vec(n, 2, records).unwrap();
        let repeated = Matrix::from_vec(repeated.len() / 2, 2, repeated).unwrap();
        let actual = frequency_covariance_with_options(
            data.view(),
            Weights::<_, Frequency>::new(&counts),
            MultivariateOptions::new(),
        )
        .unwrap()
        .1;
        let expected = sample_covariance_with_options(repeated.view(), MultivariateOptions::new())
            .unwrap()
            .1;
        for (&actual, &expected) in actual.as_slice().iter().zip(expected.as_slice()) {
            close(actual.real_component(), expected.real_component());
            close(actual.imaginary_component(), expected.imaginary_component());
        }
    }
}

#[test]
fn frequency_blocks_match_replication_in_all_scalar_domains() {
    weighted::<f32>();
    weighted::<f64>();
    weighted::<Complex<f32>>();
    weighted::<Complex<f64>>();
}
