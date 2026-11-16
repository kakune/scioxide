#![allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn eval<S: RealScalar>(
    name: &str,
    k: i32,
    z: Complex<S>,
    s: Complex<S>,
    a: Complex<S>,
) -> Result<SpecialFunctionResult<Complex<S>>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    match name {
        "w" => lambert_w_complex_with_options(z, k, LambertOptions::default()),
        "omega" => wright_omega_with_options(z, LambertOptions::default()),
        "dilog" => dilog_with_options(z, PolylogOptions::default()),
        "spence" => spence_with_options(z, PolylogOptions::default()),
        "polylog" => polylog_with_options(s, z, PolylogOptions::default()),
        "lerch" => lerch_phi_with_options(z, s, a, LerchOptions::default()),
        _ => unreachable!(),
    }
}
#[test]
fn smoke_values() {
    for (s, z) in [(0.5, 0.9), (1.0000001, 2.0), (3.5, -2.0), (-0.5, 0.9)] {
        let r = eval(
            "polylog",
            0,
            Complex::new(z, 0.),
            Complex::new(s, 0.),
            Complex::new(1., 0.),
        )
        .unwrap();

        assert!(r.termination().is_success());
    }
    assert!((lambert_w0(1.0).unwrap() - 0.5671432904097838).abs() < 1e-14);
    assert!((lambert_wm1(-0.1).unwrap() + 3.577152063957297).abs() < 1e-13);
}

#[test]
fn negative_order_continuations_and_overflow_forms() {
    for (s, z, expected) in [
        (-1., 1000., 0.001002003004005006_f64),
        (-3.5, -1e6, 0.00012774533535030672),
        (-2.5, -13.45, -0.000_167_224_592_217_053),
        (-70., -3., -1.905545903232925e63),
        (-64., 2., -2.816_838_379_668_915e99),
        (-70., 0.97, 5.424809436831935e207),
    ] {
        let r = polylog_with_options(s, z, PolylogOptions::default()).unwrap();
        let actual = r.try_value().unwrap();
        assert!(
            (actual - expected).abs()
                <= r.value().unwrap().absolute_error() + f64::EPSILON * expected.abs(),
            "s={s}, z={z}: {actual:e} != {expected:e}"
        );
    }
    let r = polylog_with_options(
        Complex::new(-200., 0.),
        Complex::new(0.5, 0.),
        PolylogOptions::default(),
    )
    .unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(
        (r.logarithmic().unwrap().value().log_magnitude().unwrap() - 936.90108422932_f64).abs()
            < 1e-11
    );
}
#[test]
fn shared_budget() {
    let r = lerch_phi_with_options(
        Complex::new(2., 1.),
        Complex::new(2., 0.5),
        Complex::new(1., 0.25),
        LerchOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
}

fn oracle_width<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut failures = Vec::new();
    let mut cases = 0;
    let input = include_str!("data/lambert_polylog.tsv");
    for line in input.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        cases += 1;
        let k = f[2].parse().unwrap();
        let number = |i: usize| S::try_from_f64_lossy(f[i].parse().unwrap()).unwrap();
        let z = Complex::new(number(3), number(4));
        let s = Complex::new(number(5), number(6));
        let a = Complex::new(number(7), number(8));
        let expected = Complex::new(number(9), number(10));
        let (rel, abs) = match (width, f[1]) {
            ("64", "w" | "omega") => (2e-12, 2e-14),
            ("64", "dilog" | "spence") => (5e-11, 5e-13),
            ("64", _) => (5e-9, 5e-12),
            ("32", "w" | "omega") => (2e-5, 2e-6),
            ("32", "dilog" | "spence") => (2e-4, 5e-6),
            _ => (1e-3, 2e-5),
        };
        match eval(f[1], k, z, s, a) {
            Ok(r) => {
                if !r.termination().is_success() {
                    failures.push(format!(
                        "{line}: {:?} work={}",
                        r.termination(),
                        r.counts().iterations()
                    ));
                    continue;
                }
                let actual = *r.value().unwrap().value();
                for (v, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                    let error = (v - e).abs().try_to_f64_exact().unwrap();
                    let limit = abs.max(rel * e.abs().try_to_f64_exact().unwrap());
                    if e != S::ZERO && e.abs() < S::MIN_POSITIVE {
                        let v = v.try_to_f64_exact().unwrap();
                        let e = e.try_to_f64_exact().unwrap();
                        let (vb, eb, limit) = if width == "32" {
                            (
                                u64::from((v as f32).to_bits()),
                                u64::from((e as f32).to_bits()),
                                16,
                            )
                        } else {
                            (v.to_bits(), e.to_bits(), 8)
                        };
                        assert!(
                            v != 0.0
                                && v.is_sign_negative() == e.is_sign_negative()
                                && vb.abs_diff(eb) <= limit,
                            "subnormal classification/ULP failure: {line}"
                        );
                    }
                    if error > limit {
                        failures.push(format!(
                            "{line}: actual={actual:?}, error={error}, limit={limit}, method={}",
                            r.algorithm().method()
                        ));
                    }
                }
                // The reported error covers the reference, allowing its rounding.
                let reported = r.value().unwrap().absolute_error();
                let rounding = S::EPSILON * S::TWO * (expected.re.abs() + expected.im.abs());
                if (actual - expected).norm() > reported + rounding {
                    failures.push(format!(
                        "{line}: actual={actual:?}, reported error {reported:?}, method={}",
                        r.algorithm().method()
                    ));
                }
            }
            Err(e) => failures.push(format!("{line}: {e:?}")),
        }
    }
    assert!(cases >= 175, "missing oracle cases");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
#[test]
fn independent_f64() {
    oracle_width::<f64>("64");
}
#[test]
fn independent_f32() {
    oracle_width::<f32>("32");
}

#[test]
fn identities_cuts_and_derivatives() {
    let one = Complex::new(1., 0.);
    for z in [
        Complex::new(0.2, 0.3),
        Complex::new(-2., 0.5),
        Complex::new(3., -1.),
    ] {
        for k in -3..=3 {
            let r = lambert_w_complex_with_options(z, k, LambertOptions::default()).unwrap();
            let w = r.try_value().unwrap();
            assert!((w * w.exp() - z).norm() < 2e-12 * z.norm());
            assert!((*r.derivative().unwrap().value() * z * (one + w) - w).norm() < 2e-12);
        }
        let s = Complex::new(2.3, 0.4);
        let a = Complex::new(1.2, 0.3);
        let phi = lerch_phi(z, s, a).unwrap();
        let shifted = lerch_phi(z, s, a + one).unwrap();
        assert!((phi - z * shifted - (-s * a.ln()).exp()).norm() < 2e-10 * phi.norm().max(1.));
        assert!((polylog(s, z).unwrap() - z * lerch_phi(z, s, one).unwrap()).norm() < 2e-12);
        assert!((spence(z).unwrap() - dilog(one - z).unwrap()).norm() < 2e-12);
        let w = wright_omega(z).unwrap();
        assert!((w + w.ln() - z).norm() < 2e-12);
    }
    for x in [1.1, 2., 10.] {
        let upper = dilog(Complex::new(x, 0.)).unwrap();
        let lower = dilog(Complex::new(x, -0.)).unwrap();
        assert!((upper - lower.conj()).norm() < 1e-13);
        assert!((upper.im - core::f64::consts::PI * x.ln()).abs() < 1e-12);
        assert!((dilog(x).unwrap() - upper.re).abs() < 1e-12);
    }
}
#[test]
fn domain_endpoints_and_strict_accuracy() {
    {
        let outcome = &(lambert_w0(f64::NAN));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    {
        let outcome = &(lambert_w0(-1.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(lambert_wm1(0.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(spence(-1.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(lerch_phi(0.5_f64, 2., -1.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(polylog(1.0_f64, 1.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(polylog(3.0_f64, 2.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    assert!(lambert_w0(-0.0_f64).unwrap().is_sign_negative());
    assert_eq!(lambert_w0(-1.0_f64 / core::f64::consts::E).unwrap(), -1.);
    assert_eq!(lambert_w0(-1.0_f32 / core::f32::consts::E).unwrap(), -1.);
    assert_eq!(
        lambert_w0_with_options(1.0, LambertOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    let r = wright_omega_with_options(-1000.0, LambertOptions::default()).unwrap();
    assert_eq!(r.try_value().unwrap(), 0.);
    assert_eq!(
        r.logarithmic().unwrap().value().log_magnitude(),
        Some(-1000.)
    );
}
#[test]
fn context_accounting_and_reused_storage() {
    use scioxide_core::{
        BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(498),
        scioxide_core::algorithm_id!("test", "special", 1),
        0,
        0,
    );
    let hundred = Budget::try_from(100).unwrap();
    // The parent limit is scope-local; the options budget bounds the child epoch.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let r = lerch_phi_scoped(
        0.9_f64,
        2.5,
        1.2,
        LerchOptions::default().with_iteration_budget(hundred),
        &mut context,
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 100);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 100);
    let complete =
        lerch_phi_scoped(0.9_f64, 2.5, 1.2, LerchOptions::default(), &mut context).unwrap();
    assert_eq!(
        complete,
        lerch_phi_with_options(0.9_f64, 2.5, 1.2, LerchOptions::default()).unwrap()
    );
    for name in ["w", "omega", "dilog", "spence", "polylog", "lerch"] {
        let r = eval(
            name,
            0,
            Complex::new(0.9_f64, 0.2),
            Complex::new(2.5, 0.1),
            Complex::new(1.2, 0.3),
        )
        .unwrap();
        assert!(r.termination().is_success());
    }
}

#[test]
fn cooperative_cancellation_precedes_work() {
    use scioxide_core::{
        CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(498),
        scioxide_core::algorithm_id!("test", "cancel", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap();
    {
        let outcome = &(polylog_scoped(2.5_f64, 0.9, PolylogOptions::default(), &mut context));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
    assert_eq!(context.counts().total().unwrap().iterations(), 0);
}

#[test]
fn specializations_and_negative_shift_sides() {
    assert!((polylog(0.0_f64, 1.).unwrap() + 0.5).abs() < 1e-14);
    assert!((lerch_phi(1.0_f64, 0., 1.2).unwrap() + 0.7).abs() < 1e-13);
    assert!((lerch_phi(0.5_f64, -2., 0.).unwrap() - 6.).abs() < 1e-11);
    let z = Complex::new(0.5, 0.);
    let s = Complex::new(2.0, 0.2);
    let lower = lerch_phi(z, s, Complex::new(-2.5, -0.)).unwrap();
    let shifted = lerch_phi(z, s, Complex::new(0.5, 0.)).unwrap();
    let mut expected = shifted * Complex::new(0.125, 0.);
    for (n, power) in [(0, 1.0), (1, 0.5), (2, 0.25)] {
        let a = Complex::new(-2.5 + f64::from(n), -0.);
        expected += (-s * a.ln()).exp() * Complex::new(power, 0.);
    }
    assert!((lower - expected).norm() < 1e-12);
    let partial = lambert_w_complex_with_options(
        Complex::new(3., 1.),
        0,
        LambertOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap();
    assert_eq!(
        partial.termination(),
        SpecialFunctionTermination::IterationLimit
    );
    assert!(partial.value().is_some());
}

#[test]
fn subnormal_real_results() {
    // exp(-100), and hence omega(-100), rounds to 27 minimum f32 subnormals.
    // The next term is exp(-200), far below half a subnormal ULP.
    assert_eq!(wright_omega(-100.0_f32).unwrap().to_bits(), 27);
    // W0(z)=z-z^2+..., so these inputs round exactly back to z.
    assert_eq!(lambert_w0(f64::from_bits(1)).unwrap().to_bits(), 1);
    assert_eq!(lambert_w0(f32::from_bits(1)).unwrap().to_bits(), 1);
}

#[test]
fn lerch_tail_does_not_stop_before_later_growth() {
    // Independent defining sums at 160/240 decimal digits; 2500 terms.
    // Early terms decrease, but a later maximum invalidates a local ratio test.
    let expected = Complex::new(-2.9990210002771456e178, -2.3515169557855314e178);
    let r = lerch_phi_with_options(
        Complex::new(0.5, 0.),
        Complex::new(-110., 100.),
        Complex::new(1., 10.),
        LerchOptions::default(),
    )
    .unwrap();
    assert!(r.termination().is_success());
    let value = r.value().unwrap();
    assert!((*value.value() - expected).norm() <= value.absolute_error());
    assert!(r.counts().iterations() > 100);
}

#[test]
fn real_lambert_branches_keep_their_identity() {
    for n in 1..100 {
        let x = -(f64::from(n) / 100.0) / core::f64::consts::E;
        let principal = lambert_w0(x).unwrap();
        let lower = lambert_wm1(x).unwrap();
        assert!(principal >= -1. && lower <= -1.);
        for w in [principal, lower] {
            assert!((w * w.exp() - x).abs() < 2e-14);
        }
    }
}

#[test]
fn real_domains_do_not_depend_on_error_tolerance() {
    let cut = 1.0_f64.next_after(2.0);
    {
        let outcome = &(polylog(3.0, cut));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(lerch_phi(cut, 2.0, 1.2));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(lerch_phi(0.5, 0.5, -0.5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(polylog(f64::NAN, 2.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    assert!((polylog(0.0, 2.).unwrap() + 2.).abs() < 1e-13);
}

#[test]
fn lerch_series_stops_at_a_valid_tail_bound_near_the_dispatch_boundary() {
    // mpmath 1.3.0 at 70 digits, for each exact argument.
    for (z, reference) in [
        (f64::from(0.95_f32), 0.982957586117605),
        (0.95_f64, 0.9829575989510685),
    ] {
        let value = polylog(5.0_f64, z).unwrap();
        assert!((value - reference).abs() <= 1e-13);
        let phi = lerch_phi(z, 5.0_f64, 1.0).unwrap();
        assert!((z * phi - reference).abs() <= 1e-13);
    }
    let value = polylog(5.0_f32, 0.95_f32).unwrap();
    assert!((value - 0.9829576_f32).abs() <= 2.0 * f32::EPSILON);
}

#[test]
fn lerch_complex_shift_uses_conditioned_continuation_on_both_lips() {
    // mpmath 1.3.0, 160/240 digits, exact binary inputs; DLMF 25.14.3_1
    // with 40 terms and arg(-log(z)) = -/+ pi. Direct lerchphi disagrees
    // on the upper lip for complex a; rotated Mellin integration agrees.
    // The former logarithmic expansion requested unstable negative-order
    // Euler--Maclaurin Hurwitz values for this complex shift (Issue #1166).
    let s = Complex::new(-0.2573900918448857, -0.3933138938466012);
    let a = Complex::new(1.2421635724634101, -0.8802614872672576);
    for (lip, expected) in [
        (0.0, Complex::new(-0.31774284070437214, 0.44987937889106067)),
        (-0.0, Complex::new(-1.3418056157463092, 0.5795018866493522)),
    ] {
        let z = Complex::new(2.494144774395604, lip);
        let result = lerch_phi_with_options(z, s, a, LerchOptions::default()).unwrap();
        assert!(result.termination().is_success());
        let value = result.value().unwrap();
        assert!(
            (*value.value() - expected).norm() <= value.absolute_error() + 2.0 * f64::EPSILON,
            "lip={lip:?} expected={expected:?} result={result:?}"
        );
        assert_eq!(lerch_phi(z, s, a).unwrap(), *value.value());
        let conjugate = lerch_phi(z.conj(), s.conj(), a.conj()).unwrap();
        assert!((conjugate - value.value().conj()).norm() <= 2.0 * value.absolute_error());
    }
}
