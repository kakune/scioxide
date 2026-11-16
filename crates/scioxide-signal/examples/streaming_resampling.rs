//! Owned-history FIR streaming and centered rational sample-rate conversion.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{
    OverlapPlan, PolyphasePlan, ResampleAlignment, SignalExecutionOptions, StreamChunk,
};
fn main() -> Result<(), SciOxideError> {
    let plan = OverlapPlan::new(&[0.25_f64, 0.5, 0.25], 4, SignalExecutionOptions::new())?;
    let mut stream = plan.stream()?;
    // Reserve once for the largest chunk; the workspace serves every later call.
    let mut work = stream.workspace(3)?;
    let mut output = vec![0.0; stream.output_len(5, StreamChunk::Final)?];
    for (chunk, kind) in [
        (&[1.0, 2.0, 3.0][..], StreamChunk::Partial),
        (&[4.0, 5.0][..], StreamChunk::Final),
    ] {
        let progress = stream.process_into(
            chunk,
            kind,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "output starting at {}: {:?}",
            progress.first_output_index(),
            &output[..progress.produced()]
        );
    }
    let resampler = PolyphasePlan::<f64>::rational_default(3, 2, ResampleAlignment::Centered)?;
    let output = resampler.resample(&[1.0, 2.0, 3.0, 4.0, 5.0])?;
    println!("3/2 centered output ({} samples): {output:?}", output.len());
    Ok(())
}
