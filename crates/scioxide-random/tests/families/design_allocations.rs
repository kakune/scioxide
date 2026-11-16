//! Warm caller-buffer contracts across designs, diagnostics and transports.
#![allow(clippy::unwrap_used)]

use scioxide_core::ExecutionContext;
use scioxide_linalg::Matrix;
use scioxide_random::*;
#[test]
fn prepared_operations_execute() {
    let options = LatinHypercubeOptions::default().with_max_proposals(8);
    let mut out = [0.0; 32];
    let mut work = SamplingWorkspace::new();
    let mut rng = SplitMix64::from_seed(42);
    let mut lhs = |rng: &mut SplitMix64, out: &mut [f64]| {
        latin_hypercube_into(
            2,
            rng,
            Rollback,
            options,
            out,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap()
    };
    // The first call grows the workspace to the design; later calls reuse it.
    lhs(&mut rng, &mut out);
    lhs(&mut rng, &mut out);
    geometric_discrepancy(&out, 2, GeometricDiscrepancy::MinimumDistance).unwrap();
    let mut cd = CenteredDiscrepancy::new(&out, 2, 17).unwrap();
    discrepancy(&out, 2, DiscrepancyMethod::Centered).unwrap();
    cd.append(&[0.5, 0.5]).unwrap();
    let poisson = PoissonDisk::with_options(
        &[0.; 2],
        &[1.; 2],
        0.1,
        16,
        PoissonDiskOptions::default()
            .with_candidates_per_point(30)
            .with_max_proposals(1000)
            .with_max_grid_cells(1000)
            .with_max_distance_checks(10_000),
    )
    .unwrap();
    let mut work = poisson.workspace().unwrap();
    poisson
        .generate_into(
            &mut rng,
            Rollback,
            &mut out,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let plan = QmcPlan::sobol(1).unwrap();
    let mut seq = plan.sequence::<f64>().unwrap();
    seq.skip_to(1).unwrap();
    let law = MultivariateNormal::new(&[0.], &Matrix::from_vec(1, 1, vec![1.]).unwrap()).unwrap();
    let gaussian = QmcMultivariateNormal::new(&law);
    let mut work = gaussian.workspace().unwrap();
    work.try_reserve(
        &SamplingPolicy::rollback(OutputPolicy::Atomic)
            .workspace_requirements(SamplingRequest::rows(32, 1))
            .unwrap(),
    )
    .unwrap();
    gaussian
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut work,
        )
        .unwrap();
    let law = Multinomial::with_options(
        1,
        &[0.5, 0.5],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut output = [0; 32];
    let transport = QmcMultinomial::new(&law).unwrap();
    let mut work = transport.workspace().unwrap();
    work.try_reserve(
        &SamplingPolicy::rollback(OutputPolicy::Atomic)
            .workspace_requirements(SamplingRequest::rows(16, 2))
            .unwrap(),
    )
    .unwrap();
    transport
        .sample_n_into(
            &mut seq,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut output,
            &mut work,
        )
        .unwrap();
}
