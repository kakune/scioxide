//! Epps–Singleton two-sample test based on empirical characteristic functions.
use super::*;

use crate::{QuantileMethod, QuantileProbability};
use scioxide_core::{ExecutionContext, ExecutionPolicy, RankThreshold, Scaled};
use scioxide_linalg::{
    DirectSolveOptions, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace,
    LuDecomposition, LuWorkspaceRequest, MatrixView, RankPolicy, VectorView, VectorViewMut,
};

struct FeatureFrame {
    origin: f64,
    frequencies: [Scaled<f64>; 2],
    native_frequencies: Option<[f64; 2]>,
}

impl FeatureFrame {
    fn new(origin: f64, semi_iqr: Scaled<f64>, points: [f64; 2]) -> Result<Self, SciOxideError> {
        let frequencies = [
            Scaled::from_value(points[0])?.div(semi_iqr)?,
            Scaled::from_value(points[1])?.div(semi_iqr)?,
        ];
        // Normal frequencies admit ordinary products without losing a quotient
        // to underflow. Extreme frequencies stay extended until the final phase.
        let native_frequencies = frequencies[0]
            .round()
            .ok()
            .zip(frequencies[1].round().ok())
            .filter(|(a, b)| a.is_normal() && b.is_normal())
            .map(|(a, b)| [a, b]);
        Ok(Self {
            origin,
            frequencies,
            native_frequencies,
        })
    }

    fn features(&self, value: f64) -> Result<[f64; 4], SciOxideError> {
        let centered = value - self.origin;
        let phases = if let Some(frequencies) = self.native_frequencies
            && centered.is_finite()
            && frequencies
                .iter()
                .all(|frequency| (frequency * centered).is_finite())
        {
            [frequencies[0] * centered, frequencies[1] * centered]
        } else {
            let centered = Scaled::from_value(value)?.sub(Scaled::from_value(self.origin)?)?;
            [
                self.frequencies[0].mul(centered)?.round()?,
                self.frequencies[1].mul(centered)?.round()?,
            ]
        };
        Ok([
            phases[0].cos(),
            phases[1].cos(),
            phases[0].sin(),
            phases[1].sin(),
        ])
    }
}

fn solve_workspace_requirements(
    execution: &LinalgExecutionOptions,
) -> Result<LinalgRequirements, SciOxideError> {
    Ok(
        LuDecomposition::<f64>::workspace_requirements(&LuWorkspaceRequest::factor(4, execution))?
            .max(&LuDecomposition::<f64>::workspace_requirements(
                &LuWorkspaceRequest::solve(4, execution),
            )?),
    )
}

fn mean_covariance<S: RealScalar>(
    values: &[S],
    frame: &FeatureFrame,
) -> Result<([f64; 4], [[f64; 4]; 4]), SciOxideError> {
    let mut mean = [0.0; 4];
    for &value in values {
        let f = frame.features(value.try_to_f64_exact()?)?;
        for i in 0..4 {
            mean[i] += f[i];
        }
    }
    for value in &mut mean {
        *value /= values.len() as f64;
    }
    let mut cov = [[0.0; 4]; 4];
    for &value in values {
        let f = frame.features(value.try_to_f64_exact()?)?;
        for i in 0..4 {
            for j in 0..4 {
                cov[i][j] += (f[i] - mean[i]) * (f[j] - mean[j]);
            }
        }
    }
    for row in &mut cov {
        for value in row {
            *value /= values.len() as f64;
        }
    }
    Ok((mean, cov))
}

/// Computes the result of [`epps_singleton_two_sample_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`epps_singleton_two_sample_test_with_options`] for the fixed default settings.
pub fn epps_singleton_two_sample_test<S: RealScalar>(
    x: &[S],
    y: &[S],
    points: [f64; 2],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    epps_singleton_two_sample_test_with_options(x, y, points, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Epps–Singleton test that two independent samples have the same
/// distribution.
///
/// The test compares the empirical characteristic functions of `x` and `y` at
/// the two frequencies in `points`, divided by the semi-interquartile range of
/// the pooled sample (Hyndman–Fan type 7 quartiles). The statistic is the
/// quadratic form of the differences of their real and imaginary parts in the
/// inverse of the estimated covariance. Its p-value is the upper tail of the
/// chi-square distribution with 4 degrees of freedom, with the small-sample
/// correction of Epps and Singleton when both samples have fewer than 25
/// observations. The usual frequencies are `[0.4, 0.8]`. Unlike rank tests, the
/// test applies to discrete data. The result does not change under translation
/// or nonzero scaling of both samples, up to rounding. The cost is
/// O(N log N) time and O(N) memory for `N = n + m`.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than five observations.
/// - `InvalidInput` if the points are not positive and distinct.
/// - `InvalidInput` if the pooled semi-interquartile range is zero.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `Singular` if the estimated covariance matrix is numerically singular.
/// - `ResourceLimit` if the scratch exceeds
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, epps_singleton_two_sample_test_with_options};
///
/// // The test is translation invariant, even for a shift of 2⁵⁰.
/// let x = [0., 1., 2., 3., 4., 5.];
/// let y = [1., 2., 3., 4., 5., 6.];
/// let test = |a: &[f64], b: &[f64]| {
///     epps_singleton_two_sample_test_with_options(a, b, [0.4, 0.8], ClassicalOptions::default())
/// };
/// let original = test(&x, &y)?;
/// let translated = test(&x.map(|v| v + 2_f64.powi(50)), &y.map(|v| v + 2_f64.powi(50)))?;
/// assert!((original.statistic() - translated.statistic()).abs() < 1e-12);
/// assert!((original.p_value() - translated.p_value()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. W. Epps and K. J. Singleton, "An omnibus test for the two-sample problem
/// using the empirical characteristic function", *J. Statist. Comput. Simul.*
/// 26 (1986), 177–203.
pub fn epps_singleton_two_sample_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    points: [f64; 2],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    sample(x, 5, scioxide_core::argument!("x"))?;
    sample(y, 5, scioxide_core::argument!("y"))?;
    if points.iter().any(|p| !p.is_finite() || *p <= 0.0) || points[0] == points[1] {
        return Err(invalid(
            scioxide_core::argument!("points"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Epps–Singleton points must be positive and distinct",
        ));
    }
    let total = x
        .len()
        .checked_add(y.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let bytes = total
        .checked_mul(core::mem::size_of::<S>())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let execution =
        LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
    let requirements = solve_workspace_requirements(&execution)?;
    let factor_bytes = LuDecomposition::<f64>::factor_storage_bytes(4)?;
    let solve_bytes = factor_bytes
        .checked_add(requirements.required_bytes::<f64>()?)
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("epps_singleton_workspace"))
        })?;
    crate::error_helpers::limit(
        bytes.max(solve_bytes),
        options.max_workspace_bytes(),
        scioxide_core::quantity!("epps_bytes"),
    )?;
    let mut pool = scioxide_core::backend::try_with_capacity(
        total,
        scioxide_core::quantity!("epps_singleton_pooled_sample"),
    )?;
    crate::error_helpers::limit(
        scioxide_core::backend::checked_mul(
            pool.capacity(),
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("epps_bytes"),
        )?,
        options.max_workspace_bytes(),
        scioxide_core::quantity!("epps_bytes"),
    )?;
    pool.extend_from_slice(x);
    pool.extend_from_slice(y);
    crate::ranking::sort_finite(&mut pool);
    // A central observation keeps Q1 <= 0 <= Q3, avoiding cancellation in
    // the semi-IQR even when a remote outlier dwarfs the central spread.
    let origin = pool[(total - 1) / 2].try_to_f64_exact()?;
    let quantile = |probability| {
        crate::empirical::quantile_scaled_sorted_by(
            pool.len(),
            QuantileProbability::new(probability)?,
            QuantileMethod::Linear,
            |index| {
                Scaled::from_value(pool[index].try_to_f64_exact()?)?
                    .sub(Scaled::from_value(origin)?)
            },
        )?
        .total()
    };
    let semi_iqr = quantile(0.75)?.sub(quantile(0.25)?)?.mul(Scaled::HALF)?;
    if semi_iqr.is_zero() {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Epps–Singleton pooled semi-IQR is zero",
        ));
    }
    // Ordering and factorization storage are never live together.
    drop(pool);
    let frame = FeatureFrame::new(origin, semi_iqr, points)?;
    let (mx, cx) = mean_covariance(x, &frame)?;
    let (my, cy) = mean_covariance(y, &frame)?;
    let n = x.len() as f64;
    let m = y.len() as f64;
    let total = n + m;
    let mut covariance = [[0.0; 4]; 4];
    let mut difference = [0.0; 4];
    for i in 0..4 {
        difference[i] = mx[i] - my[i];
        for j in 0..4 {
            covariance[i][j] = total / n * cx[i][j] + total / m * cy[i][j];
        }
    }
    let mut workspace = LinalgWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let mut factor = LuDecomposition::empty();
    LuDecomposition::factor_into(
        MatrixView::from_slice(4, 4, covariance.as_flattened())?,
        &execution.with_byte_limit(options.max_workspace_bytes() - factor_bytes),
        &mut factor,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let retained_bytes = factor.retained_bytes()?;
    let remaining_bytes = options
        .max_workspace_bytes()
        .checked_sub(retained_bytes)
        .ok_or_else(|| {
            SciOxideError::resource_limit(
                scioxide_core::quantity!("epps_bytes"),
                retained_bytes,
                options.max_workspace_bytes(),
            )
        })?;
    let controls = DirectSolveOptions::new(RankPolicy::Threshold(RankThreshold::new(
        0.0,
        64.0 * f64::EPSILON,
    )?))
    .with_execution_options(execution.with_byte_limit(remaining_bytes));
    let mut solved = [0.0; 4];
    let report = factor.solve_into(
        VectorView::from_slice(&difference),
        &controls,
        VectorViewMut::from_slice(&mut solved),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    if !report.termination().is_success() {
        return Err(SciOxideError::singular(None));
    }
    let mut statistic = finite(
        total
            * difference
                .iter()
                .zip(solved)
                .map(|(&a, b)| a * b)
                .sum::<f64>(),
    )?;
    if x.len() < 25 && y.len() < 25 {
        statistic /= 1.0 + total.powf(-0.45) + 10.1 * (n.powf(-1.7) + m.powf(-1.7));
    }
    let statistic = statistic.max(0.0);
    result(
        statistic,
        chi_tail(statistic, 4.0)?,
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        x.len() + y.len(),
        2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundaries<S: RealScalar>() {
        let convert = |value| S::try_from_f64_lossy(value).unwrap();
        let x = [0., 1., 2., 3., 4., 5.].map(convert);
        let y = [1., 2., 3., 4., 5., 6.].map(convert);
        let options = ClassicalOptions::default();
        for (a, b) in [(&x[..4], &y[..]), (&x[..], &y[..4]), (&x[..0], &y[..])] {
            assert!(
                epps_singleton_two_sample_test_with_options(a, b, [0.4, 0.8], options).is_err()
            );
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0., -1.] {
            for points in [[bad, 0.8], [0.4, bad]] {
                assert!(
                    epps_singleton_two_sample_test_with_options(&x, &y, points, options).is_err()
                );
            }
        }
        for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            let mut invalid = x;
            invalid[2] = bad;
            for (a, b) in [(&invalid, &y), (&x, &invalid)] {
                assert!(
                    epps_singleton_two_sample_test_with_options(a, b, [0.4, 0.8], options).is_err()
                );
            }
        }
        for options in [
            options.with_calibration(TestCalibration::Exact),
            options.with_alternative(TestAlternative::Greater),
            options.with_alternative(TestAlternative::Less),
        ] {
            assert!(
                epps_singleton_two_sample_test_with_options(&x, &y, [0.4, 0.8], options).is_err()
            );
        }
        let execution =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        let solve_bytes = LuDecomposition::<f64>::factor_storage_bytes(4).unwrap()
            + solve_workspace_requirements(&execution)
                .unwrap()
                .required_bytes::<f64>()
                .unwrap();
        // Exercise both the fixed solve bound and the larger pooled buffer.
        for count in [6, solve_bytes / core::mem::size_of::<S>() + 1] {
            let x: alloc::vec::Vec<_> = (0..count).map(|i| convert(i as f64)).collect();
            let y: alloc::vec::Vec<_> = (0..count).map(|i| convert(i as f64 + 1.)).collect();
            let bound = solve_bytes.max(2 * count * core::mem::size_of::<S>());
            assert!(
                epps_singleton_two_sample_test_with_options(
                    &x,
                    &y,
                    [0.4, 0.8],
                    options.with_max_workspace_bytes(bound)
                )
                .is_ok()
            );
            let error = epps_singleton_two_sample_test_with_options(
                &x,
                &y,
                [0.4, 0.8],
                options.with_max_workspace_bytes(bound - 1),
            )
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ResourceLimit { .. }
            ));
        }
        let constant = [S::ZERO; 6];
        assert!(
            epps_singleton_two_sample_test_with_options(&constant, &constant, [0.4, 0.8], options)
                .is_err()
        );
        // Nonzero IQR alone does not establish four independent features.
        let deficient = [0., 0., 0., 1., 1., 1.].map(convert);
        assert!(
            epps_singleton_two_sample_test_with_options(
                &deficient,
                &deficient,
                [0.4, 0.8],
                options
            )
            .is_err()
        );
        // Almost coincident frequencies remain numerically rank deficient.
        assert!(
            epps_singleton_two_sample_test_with_options(
                &x,
                &y,
                [0.4, f64::from_bits(0.4_f64.to_bits() + 1)],
                options
            )
            .is_err()
        );
        let overflow =
            epps_singleton_two_sample_test_with_options(&x, &y, [f64::MAX / 2., f64::MAX], options)
                .unwrap_err();
        assert!(matches!(
            overflow.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }

    #[test]
    fn checked_domains_rank_and_workspace() {
        boundaries::<f32>();
        boundaries::<f64>();
    }

    #[test]
    fn remote_outlier_does_not_erase_central_iqr() {
        for extreme in [-f64::MAX, f64::MAX] {
            let values = [extreme, 0., 1., 2., 3., 4.];
            let same = epps_singleton_two_sample_test_with_options(
                &values,
                &values,
                [0.4, 0.8],
                ClassicalOptions::default(),
            )
            .unwrap();
            assert!(same.statistic.abs() < 1e-13);
            assert!((same.p_value - 1.).abs() < 1e-13);
        }
    }

    #[test]
    fn characteristic_test_boundaries() {
        let x = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = epps_singleton_two_sample_test_with_options(
            &x,
            &y,
            [0.4, 0.8],
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!(r.statistic >= 0.0 && (0.0..=1.0).contains(&r.p_value));
        assert!(
            epps_singleton_two_sample_test_with_options(
                &x[..4],
                &y,
                [0.4, 0.8],
                ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(
            epps_singleton_two_sample_test_with_options(
                &x,
                &y,
                [0.4, 0.4],
                ClassicalOptions::default()
            )
            .is_err()
        );
        let same = epps_singleton_two_sample_test_with_options(
            &x,
            &x,
            [0.4, 0.8],
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!(same.statistic.abs() < 1e-13);
        assert!((same.p_value - 1.0).abs() < 1e-13);
        let x32 = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y32 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r32 = epps_singleton_two_sample_test_with_options(
            &x32,
            &y32,
            [0.4, 0.8],
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!((r32.statistic as f64 - r.statistic).abs() < 1e-4);
    }
}
