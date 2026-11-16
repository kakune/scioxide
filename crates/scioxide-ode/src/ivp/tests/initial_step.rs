//! The Hairer–Nørsett–Wanner initial-step heuristic.

use scioxide_core::{ErrorTolerance, NormSelection};
use scioxide_linalg::Vector;

use super::Decay;
use crate::ivp::driver::{Drive, estimate_initial_step};
use crate::ivp::workspace::IvpStorage;
use crate::stepper::testing::with_context;
use crate::stepper::{StepRequest, StepTrace, Stepper};
use crate::stiff::stage::Provider;
use crate::{AdamsMoulton, Bdf, DormandPrince5, DormandPrince853, MultistepOrder, OdeOptions};
use crate::{Rkf45, Verner65};

fn check<M: Stepper<f64, f64>>(method: M, order: usize) {
    let info = method.info();
    assert_eq!(info.initial_step_order, order, "{}", info.name);
    for (norm, normalized) in [
        (NormSelection::RootMeanSquare, 1e6),
        (NormSelection::L1, 2e6),
    ] {
        let tolerance = ErrorTolerance::scalar(1e-6, 0., norm).unwrap();
        let options = OdeOptions::new().with_error_tolerance(tolerance.clone());
        let run = Drive {
            method: &method,
            info,
            options: &options,
            tolerance: &tolerance,
            system: &Decay(2),
            events: &[],
            delay: None,
            provider: Provider::Native,
            end: 1.,
        };
        let mut storage = IvpStorage::<f64, M::Workspace, f64>::default();
        storage.reserve(&method, 2, 0, 0, false).unwrap();
        let trace = StepTrace::default();
        let request = StepRequest::new(&trace).with_error_tolerance(&tolerance);
        let initial = Vector::from_vec(vec![1., 1.]);
        let actual = with_context(|cx| {
            estimate_initial_step(&run, &mut storage, 0., &initial, 1., request, cx)
        })
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let expected = (0.01_f64 / normalized).powf(1. / order as f64);
        assert!(
            (actual - expected).abs() <= 1e-14 * expected,
            "{}: actual={actual}, expected={expected}",
            info.name
        );
    }
}

#[test]
fn initial_heuristic_uses_the_first_method_order_and_configured_norm() {
    check(Rkf45::new(), 5);
    check(DormandPrince5::new(), 5);
    check(DormandPrince853::new(), 8);
    check(Verner65::new(), 6);
    check(Bdf::new(2).unwrap(), 1);
    check(Bdf::new(5).unwrap(), 5);
    check(
        Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        1,
    );
    check(AdamsMoulton::new(3).unwrap(), 4);
    check(
        AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        1,
    );
}
