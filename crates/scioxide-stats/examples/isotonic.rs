//! Borrowed weighted axis fits and explicit block partitions.
use scioxide_core::{ExecutionContext, SciOxideError, StridedLayout};
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let values = [3.0_f64, 1.0, 4.0, 2.0, 6.0, 5.0];
    let weights = [1.0, 3.0, 1.0, 1.0, 0.0, 1.0];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2, 3], &[3, 1])?,
        Axis::Index(1),
    )?
    .with_weights(Weights::<_, Mass>::new(&weights))?;
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = IsotonicWorkspace::default();
    workspace.try_reserve(&IsotonicWorkspace::workspace_requirements(
        view, options, context,
    )?)?;
    let mut fitted = [0.0; 6];
    let mut blocks = [IsotonicBlock::default(); 6];
    let mut lanes = [IsotonicLaneResult::default(); 2];
    isotonic_regression_axes_into(
        view,
        IsotonicDirection::Increasing,
        options,
        IsotonicBatchOutput {
            fitted: &mut fitted,
            blocks: &mut blocks,
            lanes: &mut lanes,
        },
        &mut workspace,
        context,
    )?;
    assert_eq!(fitted, [1.5, 1.5, 4.0, 2.0, 2.0, 5.0]);
    for lane in lanes {
        let end = lane.block_offset + lane.diagnostics.block_count;
        println!(
            "{:?}: {:?}",
            lane.diagnostics,
            &blocks[lane.block_offset..end]
        );
    }
    Ok(())
}
