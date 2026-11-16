//! Explicit publication and rollback with a borrowed engine and a reusable workspace.

use scioxide_core::{Capacity, SciOxideError};
use scioxide_random::{
    Distribution, Normal, OutputPolicy, SamplingPolicy, SamplingRequest, SamplingWorkspace,
    SeedableRng, SplitMix64,
};

fn main() -> Result<(), SciOxideError> {
    let normal = Normal::new(0.0_f64, 1.0)?;
    let mut engine = SplitMix64::from_seed(525);
    let mut rng = &mut engine;
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let mut workspace = policy.workspace(SamplingRequest::scalars(128))?;
    let mut output = [0.0; 128];
    normal.sample_n_into(&mut rng, policy, &mut output, &mut workspace)?;
    println!("Atomic batch first sample: {}", output[0]);

    let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
    let requirements = policy.workspace_requirements(SamplingRequest::scalars(128))?;
    let mut workspace = SamplingWorkspace::new();
    assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    normal.sample_n_into(&mut rng, policy, &mut output, &mut workspace)?;
    println!(
        "Incremental scalar batch staging elements: {}",
        requirements.elements()
    );
    Ok(())
}
