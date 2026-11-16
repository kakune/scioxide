//! qmc workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::{QmcPlan, QuasiRng};

#[test]
fn test_sobol_skip_to_reproducible() {
    let sobol1_plan = QmcPlan::sobol(3).unwrap();
    let mut sobol1 = sobol1_plan.sequence::<f64>().unwrap();
    sobol1.skip(50).unwrap();
    let p1 = sobol1.next_point().unwrap();

    let sobol2_plan = QmcPlan::sobol(3).unwrap();
    let mut sobol2 = sobol2_plan.sequence::<f64>().unwrap();
    sobol2.skip_to(50).unwrap();
    let p2 = sobol2.next_point().unwrap();

    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12, "skip_to must be deterministic");
    }
}

#[test]
fn test_halton_in_unit_cube() {
    let halton_plan = QmcPlan::halton(5).unwrap();
    let mut halton = halton_plan.sequence::<f64>().unwrap();
    for _ in 0..200 {
        let p = halton.next_point().unwrap();
        for &v in &p {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
