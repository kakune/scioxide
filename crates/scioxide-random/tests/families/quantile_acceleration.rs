//! Tail, lattice, and resource regressions for approximation-seeded inverses.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_random::*;

fn continuous<D: Quantile<f64> + InverseSurvival<f64> + LogCdf<f64> + LogSurvival<f64>>(law: D) {
    let mut previous = f64::NEG_INFINITY;
    for p in [1e-12, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-3, 1.0 - 1e-12] {
        let x = law.quantile(p).unwrap();
        assert!(
            x > previous,
            "nonmonotone quantile at {p}: {x} <= {previous}"
        );
        previous = x;
        let (target, actual) = if p > 0.5 {
            ((-p).ln_1p(), law.log_sf(x).unwrap())
        } else {
            (p.ln(), law.log_cdf(x).unwrap())
        };
        assert!(
            (actual - target).abs() < 8e-8,
            "p={p}: {actual} != {target}"
        );
        let y = law.isf(p).unwrap();
        let actual = if p > 0.5 {
            law.log_cdf(y).unwrap()
        } else {
            law.log_sf(y).unwrap()
        };
        assert!(
            (actual - target).abs() < 8e-8,
            "isf p={p}: {actual} != {target}"
        );
    }
    for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(law.quantile(p).is_err());
        assert!(law.isf(p).is_err());
    }
}

#[test]
fn accelerated_quantiles_resolve_both_tails_across_shapes() {
    for (a, b) in [(0.5, 0.75), (5.0, 10.0), (50.0, 120.0)] {
        continuous(
            FisherF::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        );
        for lambda in [0.0, 2.0, 30.0] {
            continuous(
                NoncentralChiSquared::with_options(
                    a,
                    lambda,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            );
            continuous(
                NoncentralF::with_options(
                    a,
                    b,
                    lambda,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            );
        }
        for delta in [-2.0, 0.0, 2.0] {
            continuous(
                NoncentralStudentT::with_options(
                    a,
                    delta,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            );
        }
    }
    for shape in [0.0, 1.5, 10.0] {
        continuous(
            Rice::with_options(
                shape,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        );
    }
}

#[test]
fn central_limits_symmetry_and_infinite_endpoints() {
    for p in [0.0_f64, 1e-10, 0.1, 0.5, 0.9, 1.0] {
        let nc = NoncentralChiSquared::with_options(
            5.0,
            0.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let central = ChiSquared::new(5.0).unwrap();
        assert_eq!(nc.quantile(p).unwrap(), central.quantile(p).unwrap());
        assert_eq!(nc.isf(p).unwrap(), central.isf(p).unwrap());
        let nc = NoncentralF::with_options(
            5.0,
            10.0,
            0.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let central = FisherF::with_options(
            5.0,
            10.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert_eq!(nc.quantile(p).unwrap(), central.quantile(p).unwrap());
        assert_eq!(nc.isf(p).unwrap(), central.isf(p).unwrap());
        let nc = NoncentralStudentT::with_options(
            5.0,
            0.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let central = StudentT::new(5.0).unwrap();
        assert_eq!(nc.quantile(p).unwrap(), central.quantile(p).unwrap());
        assert_eq!(nc.isf(p).unwrap(), central.isf(p).unwrap());
        let rice = Rice::with_options(
            1.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let chi = NoncentralChiSquared::with_options(
            2.0,
            2.25,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert_eq!(rice.quantile(p).unwrap(), chi.quantile(p).unwrap().sqrt());
    }
    for p in [1e-10_f64, 0.1, 0.5, 0.9] {
        let x = NoncentralStudentT::with_options(
            5.0,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .quantile(p)
        .unwrap();
        let y = NoncentralStudentT::with_options(
            5.0,
            -2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .isf(p)
        .unwrap();
        assert!((x + y).abs() < 8e-8 * x.abs().max(1.0));
    }
}

#[test]
fn binary32_inverses_keep_tail_accuracy() {
    fn check<D: Quantile<f32> + InverseSurvival<f32> + LogCdf<f32> + LogSurvival<f32>>(law: D) {
        for p in [1e-20_f32, 1e-5, 0.1, 0.5, 0.9, 1.0 - 1e-5] {
            let x = law.quantile(p).unwrap();
            let y = law.isf(p).unwrap();
            let (target, lower, upper) = if p > 0.5 {
                (
                    (-p).ln_1p(),
                    law.log_sf(x).unwrap(),
                    law.log_cdf(y).unwrap(),
                )
            } else {
                (p.ln(), law.log_cdf(x).unwrap(), law.log_sf(y).unwrap())
            };
            let tolerance = 8.0 * f32::EPSILON * (1.0 + target.abs());
            assert!((lower - target).abs() <= tolerance);
            assert!((upper - target).abs() <= tolerance);
        }
    }
    check(
        FisherF::with_options(
            5.0,
            10.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralChiSquared::with_options(
            4.0,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralF::with_options(
            5.0,
            10.0,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralStudentT::with_options(
            5.0,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        Rice::with_options(
            1.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}

#[test]
fn binomial_search_returns_the_first_count_reaching_each_tail() {
    for n in [0, 1, 50, 10_000] {
        for success in [0.0, 1e-6, 0.3, 0.999_999, 1.0] {
            let law = Binomial::new(n, success).unwrap();
            for p in [1e-300_f64, 1e-12, 1e-3, 0.25, 0.5, 0.9, 1.0 - 1e-12] {
                for upper in [false, true] {
                    let k = if upper {
                        law.isf(p).unwrap()
                    } else {
                        law.quantile(p).unwrap()
                    };
                    assert!(k <= n);
                    let (target, upper) = if p > 0.5 {
                        ((-p).ln_1p(), !upper)
                    } else {
                        (p.ln(), upper)
                    };
                    let reached = |k| {
                        if upper {
                            law.log_sf(k).unwrap() <= target
                        } else {
                            law.log_cdf(k).unwrap() >= target
                        }
                    };
                    assert!(
                        reached(k),
                        "n={n}, success={success}, p={p}, upper={upper}, k={k}"
                    );
                    assert!(k == 0 || !reached(k - 1));
                }
            }
            let (low, high) = law.count_support();
            assert_eq!(law.quantile(0.0).unwrap(), low);
            assert_eq!(law.quantile(1.0).unwrap(), high.unwrap());
            assert_eq!(law.isf(0.0).unwrap(), high.unwrap());
            assert_eq!(law.isf(1.0).unwrap(), low);
        }
    }
}

#[test]
fn inverses_preserve_typed_range_and_work_failures() {
    use scioxide_core::{Budget, SciOxideErrorKind};
    let law = FisherF::with_options(
        0.5_f32,
        0.5,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(
        law.quantile(1e-30).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(
        law.isf(1e-30).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    let law = NoncentralChiSquared::with_options(
        0.5_f32,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(
        law.quantile(1e-30).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    let options = LawOptions::default().with_work_budget(Budget::ONE);
    fn exhausted<D: Quantile<f64> + InverseSurvival<f64>>(law: D) {
        assert_eq!(
            law.quantile(0.1).unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(
            law.isf(0.1).unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
    }
    exhausted(NoncentralChiSquared::with_options(4.0, 2.0, options).unwrap());
    exhausted(NoncentralF::with_options(5.0, 10.0, 2.0, options).unwrap());
    exhausted(NoncentralStudentT::with_options(5.0, 1.0, options).unwrap());
    exhausted(Rice::with_options(1.5, options).unwrap());
}

#[test]
fn noncentral_t_retains_representable_extreme_quantiles() {
    // For nu=1 the central-t start can overflow despite a smaller noncentral
    // upper quantile. The opposite shift also exercises x > MAX/2, beyond the
    // range obtained by using ln(MAX) as an asinh-coordinate bound.
    for (delta, q) in [(-1.0_f64, 1e-309), (1.0, 8e-309)] {
        let law = NoncentralStudentT::with_options(
            1.0,
            delta,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let x = law.isf(q).unwrap();
        assert!(x.is_finite() && x > 0.0);
        if delta > 0.0 {
            assert!(x > f64::MAX / 2.0);
        }
        assert!((law.log_sf(x).unwrap() - q.ln()).abs() < 8e-8);
        let mirror = NoncentralStudentT::with_options(
            1.0,
            -delta,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let y = mirror.quantile(q).unwrap();
        assert!((mirror.log_cdf(y).unwrap() - q.ln()).abs() < 8e-8);
        assert!((x + y).abs() < 8e-8 * x);
    }
}
