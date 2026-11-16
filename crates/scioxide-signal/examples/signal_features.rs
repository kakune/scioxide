//! Peaks, envelope and a delayed robust filter of a short record.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{LocalFilter, LocalFilterPlan, StreamChunk, envelope, find_peaks};
fn main() -> Result<(), SciOxideError> {
    let samples = [0.0_f64, 1.0, 1.0, 0.0, 8.0, 0.0, 0.0];
    let peaks = find_peaks(&samples)?;
    println!("peaks: {peaks:?}");
    println!("periodic analytic envelope: {:?}", envelope(&samples)?);
    // A stream emits each Hampel output once the samples after it have arrived.
    let filter = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 3. })?;
    let mut stream = filter.stream()?;
    let mut scratch = stream.workspace(4)?;
    let mut output = [0.; 7];
    let mut indices = [0; 7];
    let mut emitted = 0;
    for (chunk, kind) in [
        (&samples[..3], StreamChunk::Partial),
        (&samples[3..], StreamChunk::Final),
    ] {
        let progress = stream.process_into(
            chunk,
            kind,
            &mut output[emitted..],
            &mut indices,
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
        emitted += progress.produced();
        println!(
            "progress {progress:?}, replaced {:?}",
            &indices[..progress.replacements()]
        );
    }
    println!("streamed: {output:?}");
    println!("whole record: {:?}", filter.filter(&samples)?);
    Ok(())
}
