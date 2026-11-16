//! Work, endpoint accuracy and switch locations of the automatic stiffness-switching methods.
#![allow(clippy::unwrap_used)]
#[path = "support/switching.rs"]
mod systems;

use scioxide_core::{Budget, ErrorTolerance, EvaluationRole, NormSelection};
use scioxide_ode::*;
use systems::Problem;

fn solve<M: OdeMethod<f64> + Clone>(
    method: M,
    system: Problem,
    limit: Option<usize>,
) -> OdeSolution<f64> {
    let (absolute, relative) = system.tolerance();
    let options = OdeOptions::new().with_error_tolerance(
        ErrorTolerance::scalar(absolute, relative, NormSelection::RootMeanSquare).unwrap(),
    );
    let options = if let Some(n) = limit {
        options.with_max_steps(Budget::try_from(n).unwrap())
    } else {
        options
    };
    method
        .solve(
            &OdeProblem::new(&system, system.end()),
            &(options),
            OdeState::from_slice(0., &system.initial()).unwrap(),
        )
        .unwrap()
}

#[test]
fn switching_reuses_rhs_without_moving_switches_or_degrading_accuracy() {
    // Current-main (45b0077e2) baselines, same fixture/configuration as the manual
    // ode_switching_profile. No timing belongs in this test.
    let baseline = [
        (375458.7211259076, None),
        (5154.777282625839, None),
        (0.2308776273350949, Some(74_usize)),
        (8.926807035628947, Some(116)),
        (46.577978614548336, Some(75)),
    ];
    for (system, (old_etu, old_switch)) in Problem::ALL.into_iter().zip(baseline) {
        let solution = solve(AdamsBdf::adams_bdf().unwrap(), system, None);
        let report = solution.report();
        assert!(report.is_success(), "{system:?}: {report:?}");
        let inner = match system {
            Problem::Arenstorf | Problem::Pleiades => solve(
                AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
                system,
                None,
            ),
            _ => solve(
                Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
                system,
                None,
            ),
        };
        assert!(inner.report().is_success());
        // Diagnostics form no Jacobian: in stiff mode they reuse the inner
        // method's cached linearization.
        assert!(
            report.counts().provider(EvaluationRole::Jacobian)
                <= inner.report().counts().provider(EvaluationRole::Jacobian) + 2,
            "{system:?}: {report:?}"
        );
        assert!(
            report.counts().provider(EvaluationRole::RightHandSide) * 10
                <= inner
                    .report()
                    .counts()
                    .provider(EvaluationRole::RightHandSide)
                    * 11,
            "{system:?}: {report:?}"
        );
        let (absolute, relative) = system.tolerance();
        let reference = system.reference();
        assert_eq!(reference.len(), system.dimension());
        let etu = solution
            .state()
            .values()
            .iter()
            .zip(reference)
            .map(|(y, r)| (y - r).abs() / (absolute + relative * r.abs()))
            .fold(0_f64, f64::max);
        assert!(etu <= 1.5 * old_etu, "{system:?}: ETU={etu}");
        assert_eq!(
            report.method_switches(),
            usize::from(old_switch.is_some()),
            "{system:?}"
        );
        if let Some(old_switch) = old_switch {
            let (mut low, mut high) = (1, report.counts().proposals());
            while low < high {
                let mid = low + (high - low) / 2;
                if solve(AdamsBdf::adams_bdf().unwrap(), system, Some(mid))
                    .report()
                    .method_switches()
                    == 0
                {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            let prefix = solve(AdamsBdf::adams_bdf().unwrap(), system, Some(low));
            assert!(
                prefix.report().counts().accepted().abs_diff(old_switch) <= 2,
                "{system:?}: {:?}",
                prefix.report()
            );
            assert_eq!(
                prefix.report().last_method_switch().unwrap().reason(),
                OdeSwitchReason::Stiffness
            );
        }
    }
}
