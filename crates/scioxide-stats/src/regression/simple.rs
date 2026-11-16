//! Simple linear regression of one response on one predictor.
use super::{
    RegressionContrast, RegressionCovariance, TestAlternative, invalid, number, t_probability,
};
use crate::arithmetic::{native, native_center, native_scaled, normalized_deviation};
use crate::{ConfidenceProbability, MomentOrder, OnlineCovariance, ProbabilityPlotFit};
use scioxide_core::{RealScalar, Scaled, ScaledAccumulator, SciOxideError};

/// Holds a fitted simple linear regression `y = β₀ + β₁ x`.
///
/// The fit keeps its sums in an extended exponent range. The coefficients and
/// each inference are rounded to `S` separately, so an interval that overflows
/// `S` does not prevent reading the fitted line.
pub struct LinearRegression<S: RealScalar> {
    line: ProbabilityPlotFit<S>,
    slope: Scaled<f64>,
    intercept: Scaled<f64>,
    mean_x: ScaledAccumulator<f64>,
    predictor_squares: Scaled<f64>,
    residual_squares: Scaled<f64>,
    count: usize,
}

impl<S: RealScalar> LinearRegression<S> {
    /// Returns the slope, the intercept and the Pearson correlation.
    ///
    /// The correlation is `None` when the response is constant.
    pub fn line(&self) -> ProbabilityPlotFit<S> {
        self.line
    }

    /// Returns the t-test of `H₀: β₁ = 0` and a confidence interval for the
    /// slope.
    ///
    /// The standard error is `s / √Sxx`, where `s² = RSS / (n − 2)` is the
    /// residual variance, `Sxx = Σ (xᵢ − x̄)²` and `RSS` the residual sum of
    /// squares. The statistic `t = β̂₁ / se` has `n − 2` degrees of freedom.
    /// The interval is always two-sided at level `confidence`. `alternative`
    /// selects the tail of the p-value. Returns `None` for two observations,
    /// where `n − 2 = 0`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the standard error or an interval endpoint
    ///   overflows `S`.
    /// - The error of the Student t quantile if the critical value cannot be
    ///   computed.
    pub fn slope_inference(
        &self,
        confidence: ConfidenceProbability,
        alternative: TestAlternative,
    ) -> Result<Option<RegressionContrast<S>>, SciOxideError> {
        self.inference(false, confidence, alternative)
    }

    /// Returns the t-test of `H₀: β₀ = 0` and a confidence interval for the
    /// intercept.
    ///
    /// The standard error is `s √(1/n + x̄² / Sxx)`, with `s`, `Sxx` and the
    /// `n − 2` degrees of freedom as in [`Self::slope_inference`]. The interval
    /// is always two-sided. `alternative` selects the tail of the p-value.
    /// Returns `None` for two observations.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the standard error or an interval endpoint
    ///   overflows `S`.
    /// - The error of the Student t quantile if the critical value cannot be
    ///   computed.
    pub fn intercept_inference(
        &self,
        confidence: ConfidenceProbability,
        alternative: TestAlternative,
    ) -> Result<Option<RegressionContrast<S>>, SciOxideError> {
        self.inference(true, confidence, alternative)
    }

    fn inference(
        &self,
        intercept: bool,
        confidence: ConfidenceProbability,
        alternative: TestAlternative,
    ) -> Result<Option<RegressionContrast<S>>, SciOxideError> {
        if self.count == 2 {
            return Ok(None);
        }
        let degrees = (self.count - 2) as f64;
        let variance = self.residual_squares.div(Scaled::from_value(degrees)?)?;
        let (estimate, factor) = if intercept {
            (
                self.intercept,
                Scaled::ONE
                    .div(Scaled::from_value(self.count as f64)?)?
                    .add(self.mean_x.total()?.square()?.div(self.predictor_squares)?)?,
            )
        } else {
            (self.slope, Scaled::ONE.div(self.predictor_squares)?)
        };
        let se = variance.mul(factor)?.sqrt()?;
        let t = if se.is_zero() {
            if estimate.is_zero() {
                0.0
            } else if estimate.is_sign_negative() {
                -f64::MAX
            } else {
                f64::MAX
            }
        } else {
            let ratio = estimate.div(se)?;
            if ratio.abs_cmp(Scaled::from_value(f64::MAX)?).is_gt() {
                if ratio.is_sign_negative() {
                    -f64::MAX
                } else {
                    f64::MAX
                }
            } else {
                ratio.round()?
            }
        };
        let half = se.mul(Scaled::from_value(
            confidence.student_t_critical_value_with_options(degrees, Default::default())?,
        )?)?;
        Ok(Some(RegressionContrast {
            estimate: estimate.round_to()?,
            standard_error: se.round_to()?,
            p_value: number(t_probability(t, degrees, alternative)?)?,
            interval: scioxide_core::Interval::new(
                estimate.sub(half)?.round_to()?,
                estimate.add(half)?.round_to()?,
            )?,
            covariance: RegressionCovariance::Classical,
            degrees_of_freedom: degrees,
        }))
    }
}
/// Fits the least-squares line `y = β₀ + β₁ x` to paired observations.
///
/// The slope is `β̂₁ = Sxy / Sxx` and the intercept `β̂₀ = ȳ − β̂₁ x̄`, where
/// `Sxy = Σ (xᵢ − x̄)(yᵢ − ȳ)` and `Sxx = Σ (xᵢ − x̄)²`. The sums are
/// accumulated about the means in an extended exponent range, so data near the
/// limits of `S` do not overflow or underflow. A constant response is allowed.
/// The cost is O(n) time.
///
/// Use [`LinearRegression::slope_inference`] and
/// [`LinearRegression::intercept_inference`] for t-tests and confidence
/// intervals. For several predictors, use [`crate::PreparedRegression`].
///
/// # Errors
///
/// - `InvalidInput` if `x` and `y` differ in length or have fewer than two
///   observations.
/// - `InvalidInput` if `x` is constant.
/// - `NonFinite` if an observation is NaN or infinite.
/// - `NumericalInstability` if a coefficient overflows `S`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::linear_regression;
/// let fit = linear_regression(&[1e-200_f64, 2e-200, 3e-200], &[1e-200, 2e-200, 3e-200])?;
/// assert!((fit.line().slope - 1.0).abs() < 1e-14);
/// assert!(fit.slope_inference(Default::default(), Default::default())?.is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn linear_regression<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<LinearRegression<S>, SciOxideError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("y"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "linear regression needs paired observations",
        ));
    }
    let state = OnlineCovariance::from_slices(x, y, MomentOrder::Second)?;
    let line = crate::ProbabilityPlotFit::from_covariance(state)?;
    let snapshot = state.snapshot();
    let mx = snapshot.marginal_x().state;
    let my = snapshot.marginal_y().state;
    let (slope, intercept) = snapshot.least_squares_line()?;
    let residual_squares =
        if let Some(squares) = residual_squares(x, y, mx.mean()?, my.mean()?, slope)? {
            squares
        } else {
            let mut residual_squares = ScaledAccumulator::ZERO;
            for (&a, &b) in x.iter().zip(y) {
                let dx = crate::arithmetic::center_difference(
                    mx.mean()?,
                    ScaledAccumulator::from_value(a.into_f64())?,
                )?;
                let dy = crate::arithmetic::center_difference(
                    my.mean()?,
                    ScaledAccumulator::from_value(b.into_f64())?,
                )?;
                let residual = dy.difference_of_products(Scaled::ONE, slope, dx)?;
                residual_squares.add_scaled(residual.square()?)?;
            }
            residual_squares.total()?
        };
    Ok(LinearRegression {
        line,
        slope,
        intercept,
        mean_x: mx.mean()?,
        predictor_squares: mx.m2()?.mul(Scaled::from_value(x.len() as f64)?)?,
        residual_squares,
        count: x.len(),
    })
}

/// Returns the residual sum of squares in `f64`, or `None` when an operand
/// leaves the normal range and the extended-range pass must be used.
fn residual_squares<S: RealScalar>(
    x: &[S],
    y: &[S],
    mx: ScaledAccumulator<f64>,
    my: ScaledAccumulator<f64>,
    slope: Scaled<f64>,
) -> Result<Option<Scaled<f64>>, SciOxideError> {
    let (Some(mx), Some(my), Some(slope)) = (
        native_center(mx),
        native_center(my),
        native_scaled(slope, 64),
    ) else {
        return Ok(None);
    };
    let mut admitted = true;
    let squares = scioxide_core::compensated_reduce(x.iter().zip(y).map(|(&x, &y)| {
        let (x, y) = (x.into_f64(), y.into_f64());
        admitted &= native(x, 64) & native(y, 64);
        let dx = normalized_deviation(x, mx);
        let dy = normalized_deviation(y, my);
        // Preserve genuinely fused near-perfect-fit residuals.
        let residual = (-slope).mul_add(dx, dy);
        admitted &= native(residual, 256);
        [residual * residual]
    }));
    let Ok([squares]) = squares else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    Scaled::from_value(squares.total()?).map(Some)
}
