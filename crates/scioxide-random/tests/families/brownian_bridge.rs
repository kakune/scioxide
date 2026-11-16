//! Public workflow and allocation tests for Brownian-bridge plans.

#![allow(clippy::unwrap_used)]

use scioxide_random::Quantile;
use scioxide_random::{BrownianBridgePlan, QmcPlan, QuasiRng, SamplingWorkspace};

#[test]
fn sobol_to_normal_to_bridge_uses_public_reusable_apis() {
    const STEPS: usize = 8;

    let times: Vec<f64> = (0..=STEPS)
        .map(|index| index as f64 / STEPS as f64)
        .collect();
    let plan = BrownianBridgePlan::new(&times).unwrap();
    let sobol_plan = QmcPlan::sobol(STEPS).unwrap();
    let mut sobol = sobol_plan.sequence::<f64>().unwrap();
    sobol.skip_to(13).unwrap();
    let uniforms = sobol.next_point().unwrap();
    let normal = scioxide_random::Normal::new(0.0, 1.0).unwrap();
    let normals: Vec<f64> = uniforms
        .into_iter()
        .map(|u| normal.quantile(u))
        .collect::<Result<_, _>>()
        .unwrap();

    let mut output = vec![0.0; plan.output_len()];
    let mut workspace = SamplingWorkspace::new();
    plan.apply_into(&normals, &mut output, &mut workspace)
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));

    let negated_normals: Vec<f64> = normals.iter().map(|value| -*value).collect();
    let expected_negated: Vec<f64> = output.iter().map(|value| -*value).collect();
    plan.apply_into(&negated_normals, &mut output, &mut workspace)
        .unwrap();
    for (actual, expected) in output.iter().zip(expected_negated) {
        assert!((*actual - expected).abs() < 1.0e-14);
    }
}

#[test]
fn owned_paths_equal_caller_buffer_paths() {
    for times in [
        vec![0.0_f64, 1.0],
        vec![0.0, 0.25, 0.5, 2.0],
        (0..=37)
            .map(|i| 0.5 + f64::from(i) * f64::from(i) / 64.0)
            .collect(),
    ] {
        let plan = BrownianBridgePlan::new(&times).unwrap();
        let normals: Vec<f64> = (0..plan.input_len())
            .map(|i| (i as f64 * 0.7).sin() * 2.0)
            .collect();
        let mut output = vec![0.0; plan.output_len()];
        let mut workspace = SamplingWorkspace::new();
        plan.apply_into(&normals, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(plan.apply(&normals).unwrap(), output);
    }
    let plan = BrownianBridgePlan::new(&[0.0_f32, 1.0, 3.0]).unwrap();
    let path = plan.apply(&[1.0, 0.0]).unwrap();
    // W(3) = √3 · z[0]; W(1) is the linear interpolation at one third.
    assert!((path[1] - 3.0_f32.sqrt()).abs() < 1e-6);
    assert!((path[0] - 3.0_f32.sqrt() / 3.0).abs() < 1e-6);
}
