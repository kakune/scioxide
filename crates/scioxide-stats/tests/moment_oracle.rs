//! Exact-rational, independently generated moment references and fixed error envelopes.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/reductions.rs"]
mod reduction_test;
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

fn decode(value: &str, scalar: &str) -> f64 {
    if scalar == "f32" {
        f64::from(f32::from_bits(u32::from_str_radix(value, 16).unwrap()))
    } else {
        f64::from_bits(u64::from_str_radix(value, 16).unwrap())
    }
}
fn values<S: RealScalar>(encoded: &str, scalar: &str) -> Vec<S> {
    if encoded == "-" {
        Vec::new()
    } else {
        encoded
            .split(',')
            .map(|v| S::try_from_f64_lossy(decode(v, scalar)).unwrap())
            .collect()
    }
}
// Exercise observations, complete slices, and serial/balanced partition trees.
fn accumulate<A: Copy>(
    length: usize,
    path: usize,
    build: impl Fn(core::ops::Range<usize>, bool) -> Result<A, SciOxideError>,
    merge: impl Fn(&mut A, &A) -> Result<(), SciOxideError>,
) -> Result<A, SciOxideError> {
    if path < 2 || length == 0 {
        return build(0..length, path == 1);
    }
    let chunk = path - 1;
    let mut level = (0..length)
        .step_by(chunk)
        .map(|start| build(start..(start + chunk).min(length), path >= 4))
        .collect::<Result<Vec<_>, _>>()?;
    if path < 4 {
        return level[1..].iter().try_fold(level[0], |mut left, right| {
            merge(&mut left, right)?;
            Ok(left)
        });
    }
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                let mut left = pair[0];
                if let Some(right) = pair.get(1) {
                    merge(&mut left, right)?;
                }
                Ok(left)
            })
            .collect::<Result<_, SciOxideError>>()?;
    }
    Ok(level[0])
}

fn weighted_covariance<S: RealScalar, M: MomentWeights>(
    metric: &str,
    values: &[S],
    ys: &[S],
    weights: &[S],
    path: usize,
) -> Result<Option<S>, SciOxideError> {
    let a = accumulate(
        values.len(),
        path,
        |range, _| {
            let mut a = OnlineWeightedCovariance::<S, M>::new(MomentOrder::Fourth);
            for i in range {
                a.observe(values[i], ys[i], weights[i])?;
            }
            Ok(a)
        },
        OnlineWeightedCovariance::merge,
    )?
    .snapshot();
    match metric {
        "covariance" => a.population_covariance(),
        "sample_covariance" => a.sample_covariance(),
        _ => a.correlation(),
    }
}
fn weighted_moments<S: RealScalar, M: MomentWeights>(
    values: &[S],
    weights: &[S],
    path: usize,
) -> Result<WeightedMomentsSnapshot<S, M>, SciOxideError> {
    accumulate(
        values.len(),
        path,
        |range, slice| {
            if slice {
                return OnlineWeightedMoments::from_slices(
                    &values[range.clone()],
                    Weights::<_, M>::new(&weights[range]),
                    MomentOrder::Fourth,
                );
            }
            let mut a = OnlineWeightedMoments::<S, M>::new(MomentOrder::Fourth);
            for i in range {
                a.observe(values[i], weights[i])?;
            }
            Ok(a)
        },
        OnlineWeightedMoments::merge,
    )
    .map(|a| a.snapshot())
}
/// Model-independent weighted metrics; model-specific estimators return `None`.
fn weighted_metric<S: RealScalar, M: MomentWeights>(
    a: WeightedMomentsSnapshot<S, M>,
    metric: &str,
) -> Option<Result<Option<S>, SciOxideError>> {
    Some(match metric {
        "mean" => a.mean(),
        "m2" => a.central_moment(2),
        "m3" => a.central_moment(3),
        "m4" => a.central_moment(4),
        "skew" => a.skewness(BiasCorrection::None),
        "kurtosis" => a.kurtosis(BiasCorrection::None, KurtosisConvention::Fisher),
        "skew_corrected" => a.skewness(BiasCorrection::Corrected),
        "kurtosis_corrected" => a.kurtosis(BiasCorrection::Corrected, KurtosisConvention::Fisher),
        "effective" => a.effective_sample_size(),
        "degrees" => a.degrees_of_freedom(),
        "sample" => a.sample_variance(),
        "meanvar" => a.variance_of_weighted_mean(),
        _ => return None,
    })
}

fn query<S: RealScalar + NumericScalar<Real = S>>(
    mode: &str,
    metric: &str,
    values: &[S],
    weights: &[S],
    path: usize,
) -> Result<Option<S>, SciOxideError> {
    let view_shape = [1, values.len()];
    let view_strides = [values.len(), 1];
    let view = BatchView::new(
        values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new().with_moment_order(MomentOrder::Fourth);
    if let Some(order) = metric.strip_prefix("batch") {
        return reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            order.parse().unwrap(),
            None,
            options,
            ExecutionContext::sequential(),
        )
        .map(|v| v[0]);
    }
    if ["covariance", "sample_covariance", "correlation"].contains(&metric) {
        let ys = values.iter().map(|&x| -x).collect::<Vec<_>>();
        if path == 6 && metric == "sample_covariance" {
            let pairs = values
                .iter()
                .zip(&ys)
                .flat_map(|(&x, &y)| [x, y])
                .collect::<Vec<_>>();
            let matrix = MatrixView::from_slice(values.len(), 2, &pairs)?;
            let (_, covariance) = if mode == "unit" {
                scioxide_stats::sample_covariance_with_options(matrix, MultivariateOptions::new())?
            } else {
                {
                    let counts = weights
                        .iter()
                        .map(|&w| w.try_to_usize_exact().map(|n| n as u64))
                        .collect::<Result<Vec<_>, _>>()?;
                    scioxide_stats::frequency_covariance_with_options(
                        matrix,
                        Weights::<_, Frequency>::new(&counts),
                        MultivariateOptions::new(),
                    )?
                }
            };
            return covariance.view().get(0, 1).map(Some);
        }
        if mode == "unit" {
            let a = accumulate(
                values.len(),
                path,
                |range, slice| {
                    if slice {
                        return OnlineCovariance::from_slices(
                            &values[range.clone()],
                            &ys[range],
                            MomentOrder::Fourth,
                        );
                    }
                    let mut a = OnlineCovariance::new(MomentOrder::Fourth);
                    for i in range {
                        a.observe(values[i], ys[i])?;
                    }
                    Ok(a)
                },
                OnlineCovariance::merge,
            )?
            .snapshot();
            return match metric {
                "covariance" => a.population_covariance(),
                "sample_covariance" => a.sample_covariance(),
                _ => a.correlation(),
            };
        }
        return match mode {
            "reliability" => {
                weighted_covariance::<S, Reliability>(metric, values, &ys, weights, path)
            }
            "analytic" => weighted_covariance::<S, Analytic>(metric, values, &ys, weights, path),
            _ => weighted_covariance::<S, Frequency>(metric, values, &ys, weights, path),
        };
    }
    match mode {
        "frequency" => {
            let a = weighted_moments::<S, Frequency>(values, weights, path)?;
            return weighted_metric(a, metric).unwrap_or_else(|| {
                let order = metric.strip_prefix('k').unwrap();
                a.k_statistic(order.parse().unwrap())
            });
        }
        "reliability" => {
            let a = weighted_moments::<S, Reliability>(values, weights, path)?;
            return weighted_metric(a, metric)
                .unwrap_or_else(|| panic!("unknown weighted metric {metric}"));
        }
        "analytic" => {
            let a = weighted_moments::<S, Analytic>(values, weights, path)?;
            return weighted_metric(a, metric).unwrap_or_else(|| {
                assert_eq!(metric, "noise", "unknown weighted metric {metric}");
                a.noise_scale()
            });
        }
        _ => {}
    }
    let a = if mode == "trim" {
        reduction_test::trimmed(
            view.with_axis(options_axis).unwrap(),
            TrimLimits::new(Some(S::ZERO), Some(S::try_from_f64_lossy(4.0)?))?,
            options,
            ExecutionContext::sequential(),
        )?[0]
    } else {
        accumulate(
            values.len(),
            path,
            |range, slice| {
                if slice {
                    return OnlineMoments::from_slice(&values[range], MomentOrder::Fourth);
                }
                let mut a = OnlineMoments::new(MomentOrder::Fourth);
                for &x in &values[range] {
                    a.observe(x)?;
                }
                Ok(a)
            },
            OnlineMoments::merge,
        )?
        .snapshot()
    };
    match metric {
        "mean" | "tmean" => a.mean(),
        "minimum" | "tmin" => Ok(a.minimum()),
        "maximum" | "tmax" => Ok(a.maximum()),
        "m2" => a.central_moment(2),
        "m3" => a.central_moment(3),
        "m4" => a.central_moment(4),
        "skew" => a.skewness(BiasCorrection::None),
        "kurtosis" => a.kurtosis(BiasCorrection::None, KurtosisConvention::Fisher),
        "skew_corrected" => a.skewness(BiasCorrection::Corrected),
        "kurtosis_corrected" => a.kurtosis(BiasCorrection::Corrected, KurtosisConvention::Fisher),
        "variance" | "tvar" => a.variance(1),
        "std" | "tstd" => a.standard_deviation(1),
        "sem" | "tsem" => a.standard_error(1),
        "variation" => a.coefficient_of_variation(0),
        k if k.starts_with("kvar") => a.k_statistic_variance(k[4..].parse().unwrap()),
        k if k.starts_with('k') => a.k_statistic(k[1..].parse().unwrap()),
        _ => panic!("unknown metric {metric}"),
    }
}
fn check<S: RealScalar + NumericScalar<Real = S>>(fields: &[&str], path: usize) {
    let xs = values::<S>(fields[5], fields[1]);
    let ws = values::<S>(fields[6], fields[1]);
    let result = query(fields[2], fields[3], &xs, &ws, path);
    if fields[7] == "none" {
        assert_eq!(
            result.unwrap(),
            None,
            "{} {} {}",
            fields[0],
            fields[2],
            fields[3]
        );
        return;
    }
    if fields[7] == "overflow" {
        assert!(
            matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { .. })),
            "expected numerical range error: {fields:?}, {result:?}"
        );
        return;
    }
    let actual = result.unwrap().unwrap().try_to_f64_exact().unwrap();
    let expected = decode(fields[7], fields[1]);
    let degree: i32 = fields[4].parse().unwrap();
    let data = xs
        .iter()
        .map(|x| x.try_to_f64_exact().unwrap())
        .collect::<Vec<_>>();
    let magnitude = data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let range = data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - data.iter().copied().fold(f64::INFINITY, f64::min);
    if actual == 0.0 && expected == 0.0 {
        return;
    }
    // Centered diagnostics are scaled only by spread, never the raw offset.
    // Normalize powers individually to avoid overflowing the test envelope.
    let radius = if range > 0.0 && range.is_finite() {
        range
    } else {
        1.0
    };
    let normalize = |value: f64| (0..degree).fold(value, |v, _| v / radius);
    let scale =
        if ["mean", "tmean", "minimum", "maximum", "tmin", "tmax", "k1"].contains(&fields[3]) {
            normalize(magnitude).max(1.0)
        } else {
            normalize(expected.abs()).max(1.0)
        };
    let residual = (normalize(actual) - normalize(expected)).abs();
    let (epsilon, quantum) = if fields[1] == "f32" {
        (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
    } else {
        (f64::EPSILON, f64::from_bits(1))
    };
    let bound = 512.0 * epsilon * scale + normalize(quantum);
    assert!(bound.is_finite(), "nonfinite oracle envelope");
    assert!(
        actual.is_finite() && residual <= bound,
        "{} {} {}: {actual} != {expected}; bound {bound}",
        fields[0],
        fields[2],
        fields[3]
    );
}
#[test]
fn exact_binary_inputs_match_independent_centered_rational_references() {
    let mut count = 0;
    for row in include_str!("data/moments.tsv")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields = row.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 9);
        for path in 0..6 {
            if fields[1] == "f32" {
                check::<f32>(&fields, path);
            } else {
                check::<f64>(&fields, path);
            }
        }
        count += 1;
    }
    assert_eq!(count, 700);
}

#[test]
fn adjacent_translated_and_repeated_inputs_match_exact_rationals() {
    for row in include_str!("data/centered_moments.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let fields = row.split('\t').collect::<Vec<_>>();
        for path in 0..6 {
            if fields[1] == "f32" {
                check::<f32>(&fields, path);
            } else {
                check::<f64>(&fields, path);
            }
        }
    }
}

#[test]
fn multivariate_covariance_matches_independent_centered_rational_references() {
    for row in include_str!("data/centered_moments.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let fields = row.split('\t').collect::<Vec<_>>();
        if fields[3] != "sample_covariance" || !["unit", "frequency"].contains(&fields[2]) {
            continue;
        }
        if fields[1] == "f32" {
            check::<f32>(&fields, 6);
        } else {
            check::<f64>(&fields, 6);
        }
    }
}
