//! Design, stream and zero-phase filter a sampled record.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{
    IirBand, IirDomain, IirFamily, IirFilter, IirRequest, IirState, SignalExecutionOptions,
    SignalLayout, design_iir,
};
fn main() -> Result<(), SciOxideError> {
    let design = design_iir(
        6,
        IirFamily::Butterworth,
        IirBand::Lowpass(0.2_f64),
        IirDomain::Digital,
    )?;
    let sos = design.to_sos()?;
    let mut input = [0.; 128];
    input[32] = 1.;
    // Stream the record in two chunks; the state carries the delays across them.
    let plan = IirFilter::from_sos(&sos, SignalExecutionOptions::new())?;
    let mut streamed = [0.; 128];
    let mut state = IirState::new(&plan, 1)?;
    let mut work = plan.workspace(IirRequest::Filter(SignalLayout::Contiguous(64)))?;
    for (x, y) in input.chunks(64).zip(streamed.chunks_mut(64)) {
        plan.filter_into(
            x,
            y,
            &mut state,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
    }
    let causal = sos.filter(&input)?;
    let zero_phase = sos.forward_backward(&input)?;
    println!(
        "{} sections, pole margin {}, streamed equals one call: {}, zero-phase center {}",
        sos.sections().len(),
        design.stability_margin(),
        streamed[..] == causal[..],
        zero_phase[32]
    );
    Ok(())
}
