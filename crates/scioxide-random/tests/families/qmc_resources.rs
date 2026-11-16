//! Resource contracts for qmc.
#![allow(clippy::unwrap_used)]

use scioxide_random::{OutputPolicy, QuasiRng, SamplingPolicy};
#[path = "../support/qmc_plans.rs"]
mod fixtures;
use fixtures::plans;

#[test]
fn seeking_and_prepared_output_and_export_execute() {
    for plan in plans() {
        let mut cursor = plan.sequence::<f64>().unwrap();
        let mut output = [0.0; 48];
        let batch = SamplingPolicy::rollback(OutputPolicy::Atomic);
        let mut workspace = batch
            .workspace(scioxide_random::SamplingRequest::scalars(48))
            .unwrap();
        let mut checkpoint = vec![0; cursor.state_bytes().unwrap()];
        (|| {
            cursor.skip_to(1 << 40)?;
            cursor.points_into(batch, &mut output, &mut workspace)?;
            cursor.export_state_into(&mut checkpoint)
        })()
        .unwrap();
    }
}
