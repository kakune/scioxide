//! Linear contrasts, robust slopes and simultaneous group comparisons.
use scioxide_core::SciOxideError;
use scioxide_linalg::MatrixView;
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let x = [0., 1., 2., 3., 4.];
    let y = [1., 2., 4., 4., 5.];
    let design = [1., 0., 1., 1., 1., 2., 1., 3., 1., 4.];
    let model = PreparedRegression::new(
        MatrixView::from_slice(5, 2, &design)?,
        Unweighted,
        LinearRegressionOptions::default().with_max_bytes(1_000_000),
    )?;
    let fit = ordinary_least_squares(MatrixView::from_slice(5, 2, &design)?, &y)?;
    let tests = model.coefficient_inference(
        &y,
        RegressionCovariance::Classical,
        ConfidenceProbability::default(),
    )?;
    let slope = &tests[1];
    println!(
        "coefficients={:?}, rank={}, residual={}, slope CI={:?}",
        fit.coefficients().as_slice(),
        model.rank().rank(),
        fit.residual_norm(),
        slope.interval()
    );
    let robust = robust_slope(&x, &y, RobustSlopeMethod::TheilSen)?;
    println!(
        "Theil-Sen slope={}, intercept={}",
        robust.slope(),
        robust.intercept()
    );
    let a = [1., 2., 3.];
    let b = [2., 3., 4.];
    let groups = PreparedGroups::new(&[&a, &b], StatisticsOptions::new().with_max_bytes(10_000))?;
    let anova = groups.anova(GroupTestMethod::Anova)?;
    let comparisons = groups.pairwise(PostHocMethod::TukeyKramer)?;
    println!(
        "F={}, p={}, adjusted p={}",
        anova.statistic(),
        anova.p_value(),
        comparisons.comparisons()[0].p_value()
    );
    Ok(())
}
