//! Pole placement on routine controllable systems, conjugate canonicalization
//! and solve-scope accounting.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, ExecutionContext, SciOxideErrorView};
use scioxide_linalg::Matrix;
use scioxide_signal::{LtiTime, PolePlacementOptions, PolePlacementReport, StateSpace};

fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

/// Deterministic uniform samples in `[-1, 1)`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn system(a: Vec<f64>, b: Vec<f64>, n: usize, m: usize) -> StateSpace<f64> {
    let a = Matrix::from_vec(n, n, a).unwrap();
    let b = Matrix::from_vec(n, m, b).unwrap();
    let c = Matrix::identity(n).unwrap();
    let d = Matrix::zeros(n, m).unwrap();
    StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap()
}

#[test]
fn random_siso_systems_place_poles_to_their_eigenvector_conditioning() {
    // SISO gains are unique, so accuracy is limited by the condition of the
    // eigenvector matrix V.
    let mut random = Lcg(2024);
    for n in 4..=10 {
        let mut requested = Vec::new();
        for k in 0..n / 2 {
            let re = -1.0 - 0.5 * k as f64;
            requested.extend([Complex::new(re, 1.0), Complex::new(re, -1.0)]);
        }
        if n % 2 == 1 {
            requested.push(Complex::new(-0.7, 0.0));
        }
        for _ in 0..20 {
            let a: Vec<f64> = (0..n * n).map(|_| random.next()).collect();
            let b: Vec<f64> = (0..n).map(|_| random.next()).collect();
            let norm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let (a_norm, b_norm) = (norm(&a), norm(&b));
            let report = system(a, b, n, 1)
                .place_poles_with_options(&requested, PolePlacementOptions::new())
                .unwrap();
            // First-order eigenvalue sensitivity of the rounded closed loop
            // A - B K: cond(V) epsilon (|A| + |B| |K|).
            let condition = 1.0 / report.transfer_condition().unwrap().reciprocal().value();
            let scale = 1.0 + a_norm + b_norm * norm(report.gain().as_slice());
            let bound = 64.0 * f64::EPSILON * condition * scale;
            assert!(
                report.maximum_residual() <= bound,
                "n={n}: residual {:e}, bound {bound:e}",
                report.maximum_residual()
            );
            // One null-space column per pole: a single deterministic attempt.
            assert_eq!(report.counts().own().iterations(), 1);
            if n <= 6 {
                assert!(
                    report.is_success(),
                    "n={n}: {:e}",
                    report.maximum_residual()
                );
            }
            assert!(report.gain().as_slice().iter().all(|k| k.is_finite()));
        }
    }
}

#[test]
fn nearly_conjugate_requests_are_canonicalized_before_placement() {
    let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -2.0, -3.0];
    let b = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mimo = system(a.clone(), b, 3, 2);
    let siso = system(a, vec![0.0, 0.0, 1.0], 3, 1);
    for perturbation in [1e-15, -2e-15] {
        let requested = [
            Complex::new(-1.0, 2.0),
            Complex::new(-1.0 + perturbation, -2.0 * (1.0 + perturbation)),
            Complex::new(-3.0, 0.0),
        ];
        for model in [&mimo, &siso] {
            let report = model
                .place_poles_with_options(&requested, PolePlacementOptions::new())
                .unwrap();
            assert!(report.is_success(), "{:e}", report.maximum_residual());
            let poles = report.requested_poles();
            assert_eq!(poles[0], Complex::new(-3.0, 0.0));
            assert_eq!(poles[1], poles[2].conj());
        }
    }
    // A pole within rounding of the real axis and without a partner is real.
    let report = siso
        .place_poles_with_options(
            &[
                Complex::new(-1.0, 1e-17),
                Complex::new(-2.0, 0.0),
                Complex::new(-3.0, 0.0),
            ],
            PolePlacementOptions::new(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!(report.requested_poles().iter().all(|p| p.im == 0.0));
    // A genuine lone complex pole and an unmatched multiplicity stay invalid.
    for requested in [
        [
            Complex::new(-1.0, 1.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-3.0, 0.0),
        ],
        [
            Complex::new(-1.0, 1.0),
            Complex::new(-1.0, 1.0),
            Complex::new(-1.0, -1.0),
        ],
    ] {
        let error = mimo
            .place_poles_with_options(&requested, PolePlacementOptions::new())
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("requested")
        ));
    }
}

#[test]
fn placement_runs_in_the_callers_scope() {
    use scioxide_core::{
        CancellationToken, ExecutionStopReason, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let model = system(
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -2.0, -3.0],
        vec![0.0, 0.0, 1.0],
        3,
        1,
    );
    let requested = [
        Complex::new(-1.0, 1.0),
        Complex::new(-1.0, -1.0),
        Complex::new(-5.0, 0.0),
    ];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            cx(),
            SolveIdentity::new(
                ProblemToken::new(434),
                scioxide_core::algorithm_id!("signal", "placement_test", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let report = model
        .place_poles_scoped(&requested, PolePlacementOptions::new(), &mut context)
        .unwrap();
    assert!(report.is_success());
    assert_eq!(context.counts().own().iterations(), 0);
    let total = report.counts().total().unwrap().iterations();
    assert!(total > 0);
    assert_eq!(context.counts().nested().iterations(), total);
    // The owned placement is the scoped one in a fresh native root scope.
    let one_shot = model.place_poles(&requested).unwrap();
    let scoped = model
        .place_poles_scoped(&requested, PolePlacementOptions::new(), cx())
        .unwrap();
    let bits = |report: &PolePlacementReport<f64>| -> Vec<u64> {
        let poles = report.achieved_poles().iter().flat_map(|p| [p.re, p.im]);
        report
            .gain()
            .as_slice()
            .iter()
            .copied()
            .chain(poles)
            .map(f64::to_bits)
            .collect()
    };
    assert_eq!(bits(&one_shot), bits(&scoped));

    let token = CancellationToken::new();
    token.cancel();
    let error = model
        .place_poles_scoped(
            &requested,
            PolePlacementOptions::new(),
            cx().with_cancellation(&token),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::Cancelled
        }
    ));
}
