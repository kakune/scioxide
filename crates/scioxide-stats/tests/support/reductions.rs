#![allow(dead_code)]
//! Test conveniences that exercise reserved reductions with injected capabilities.
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

pub fn moments<S: RealScalar>(
    view: BatchView<'_, S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        OnlineMoments::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_moments_into(view, options, &mut output, &mut ws, cx)?;
    Ok(output)
}
pub fn trimmed<S: RealScalar>(
    view: BatchView<'_, S>,
    limits: TrimLimits<S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        OnlineMoments::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_trimmed_moments_into(view, limits, options, &mut output, &mut ws, cx)?;
    Ok(output)
}
pub fn covariance<S: RealScalar>(
    view: BatchView<'_, S>,
    y: BatchView<'_, S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<CovarianceSnapshot<S>>, SciOxideError> {
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        OnlineCovariance::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_covariance_into(view, y, options, &mut output, &mut ws, cx)?;
    Ok(output)
}
pub fn power<S: RealScalar>(
    view: BatchView<'_, S>,
    order: usize,
    center: Option<S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<Option<S>>, SciOxideError> {
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        None,
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_moment_into(view, order, center, options, &mut output, &mut ws, cx)?;
    Ok(output)
}
fn weights<S: RealScalar>(
    view: BatchView<'_, S>,
    weights: MatrixView<'_, S>,
) -> Result<Vec<S>, SciOxideError> {
    if view.layout().shape().extents() != [weights.rows(), weights.cols()] {
        return Err(SciOxideError::dimension_mismatch(
            view.layout().shape().extents(),
            [weights.rows(), weights.cols()],
        ));
    }
    let mut output = scioxide_core::backend::try_with_capacity(
        view.batches() * view.samples(),
        scioxide_core::quantity!("test_weights"),
    )?;
    for batch in 0..view.batches() {
        for i in 0..view.samples() {
            let (r, c) = match view.axis() {
                Axis::All => (i / weights.cols(), i % weights.cols()),
                Axis::Index(0) => (i, batch),
                _ => (batch, i),
            };
            output.push(weights.get(r, c)?);
        }
    }
    Ok(output)
}
/// Attach real test weights to a view under a typed sampling model. Frequency
/// counts are converted exactly to the `u64` batch representation.
pub trait TestModel: MomentWeights {
    fn attach<'a, S: RealScalar>(
        view: BatchView<'a, S>,
        weights: &'a [S],
        counts: &'a [u64],
    ) -> Result<BatchView<'a, S, Self>, SciOxideError>;
}
impl TestModel for Frequency {
    fn attach<'a, S: RealScalar>(
        view: BatchView<'a, S>,
        _: &'a [S],
        counts: &'a [u64],
    ) -> Result<BatchView<'a, S, Self>, SciOxideError> {
        view.with_weights(Weights::<_, Frequency>::new(counts))
    }
}
impl TestModel for Reliability {
    fn attach<'a, S: RealScalar>(
        view: BatchView<'a, S>,
        weights: &'a [S],
        _: &'a [u64],
    ) -> Result<BatchView<'a, S, Self>, SciOxideError> {
        view.with_weights(Weights::<_, Reliability>::new(weights))
    }
}
impl TestModel for Analytic {
    fn attach<'a, S: RealScalar>(
        view: BatchView<'a, S>,
        weights: &'a [S],
        _: &'a [u64],
    ) -> Result<BatchView<'a, S, Self>, SciOxideError> {
        view.with_weights(Weights::<_, Analytic>::new(weights))
    }
}
fn counts<S: RealScalar, M: TestModel>(weights: &[S]) -> Result<Vec<u64>, SciOxideError> {
    if M::SEMANTICS != WeightSemantics::Frequency {
        return Ok(Vec::new());
    }
    weights
        .iter()
        .map(|&w| w.try_to_usize_exact().map(|n| n as u64))
        .collect()
}
pub fn weighted_moments<S: RealScalar, M: TestModel>(
    view: BatchView<'_, S>,
    w: MatrixView<'_, S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<WeightedMomentsSnapshot<S, M>>, SciOxideError> {
    let weights = weights(view, w)?;
    let counts = counts::<S, M>(&weights)?;
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        OnlineWeightedMoments::<S, M>::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_weighted_moments_into(
        M::attach(view, &weights, &counts)?,
        options,
        &mut output,
        &mut ws,
        cx,
    )?;
    Ok(output)
}
pub fn weighted_covariance<S: RealScalar, M: TestModel>(
    view: BatchView<'_, S>,
    y: BatchView<'_, S>,
    w: MatrixView<'_, S>,
    options: StatisticsOptions,
    cx: ExecutionContext<'_>,
) -> Result<Vec<WeightedCovarianceSnapshot<S, M>>, SciOxideError> {
    let weights = weights(view, w)?;
    let counts = counts::<S, M>(&weights)?;
    let req = ReductionWorkspace::workspace_requirements(view, options, cx)?;
    let mut ws = ReductionWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        view.batches(),
        OnlineWeightedCovariance::<S, M>::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("test_output"),
    )?;
    reduce_weighted_covariance_into(
        M::attach(view, &weights, &counts)?,
        y,
        options,
        &mut output,
        &mut ws,
        cx,
    )?;
    Ok(output)
}
