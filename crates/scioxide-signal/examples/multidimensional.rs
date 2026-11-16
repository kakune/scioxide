//! Physical-coordinate monogenic vector and single-orthant analytic signal.
use scioxide_core::{NdView, SciOxideError, StridedLayout};
use scioxide_signal::{AnalyticSignalNdPlan, RieszPlan, SignalExecutionOptions};
fn main() -> Result<(), SciOxideError> {
    let shape = [8, 8];
    let strides = [8, 1];
    let samples: Vec<f64> = (0..64)
        .map(|i| (core::f64::consts::TAU * ((i / 8 + i % 8) as f64) / 8.).cos())
        .collect();
    let layout = StridedLayout::new(&shape, &strides)?;
    let options = SignalExecutionOptions::new();
    let analytic = AnalyticSignalNdPlan::new(&shape, &[0, 1], options)?
        .analytic(NdView::new(&samples, layout)?)?;
    let plan = RieszPlan::new(&shape, &[0, 1], &[1., 2.], options)?;
    let tuple = plan.monogenic(NdView::new(&samples, layout)?)?;
    let index = 1;
    let (f, rx, ry) = (tuple[index], tuple[64 + index], tuple[128 + index]);
    println!("analytic sample: {:?}", analytic[index]);
    println!(
        "monogenic shape: {:?}; amplitude: {}; phase: {}",
        plan.monogenic_shape(),
        f.hypot(rx.hypot(ry)),
        rx.hypot(ry).atan2(f)
    );
    Ok(())
}
