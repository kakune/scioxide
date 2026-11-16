//! Independent principal Legendre continuation with complex parameters.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn references<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/complex_elliptic.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap();
        let c = |i| {
            Complex::new(
                S::try_from_f64_lossy(n(i)).unwrap(),
                S::try_from_f64_lossy(n(i + 1)).unwrap(),
            )
        };
        let a = c(2);
        let m = c(4);
        let options = EllipticOptions::default();
        let r = match f[1] {
            "k" => elliptic_k_with_options(m, options),
            "e" => elliptic_e_with_options(m, options),
            "f" => elliptic_f_with_options(a, m, options),
            "ei" => elliptic_e_inc_with_options(a, m, options),
            _ => elliptic_pi_with_options(a, m, options),
        }
        .unwrap_or_else(|error| panic!("{row}: {error:?}"));
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let s = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(s, f[8].parse().unwrap());
        let expected = Complex::new(n(6), n(7));
        // The reported scaled error covers the reference, allowing its rounding.
        let scaled = r.scaled().unwrap();
        let reported = scaled_reference::error_at_exponent(
            scaled.value(),
            scaled.absolute_error().into_f64(),
            f[8].parse().unwrap(),
        );
        let precision = if width == "32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        assert!(
            (actual - expected).norm() <= reported + 2.0 * precision * expected.norm(),
            "{row}: reported scaled error {reported:e}"
        );
        assert!(
            (actual - expected).norm() < if width == "32" { 3e-6 } else { 5e-12 },
            "{row}: {actual:?} expected={expected:?}"
        );
    }
}
#[test]
fn independent_f64() {
    references::<f64>("64");
}
#[test]
fn independent_f32() {
    references::<f32>("32");
}

#[test]
fn branch_identities_and_scoped_resources() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let m = Complex::new(0.5, 0.25);
    let phi = Complex::new(0.25, 0.1);
    let one = Complex::new(1.0, 0.0);
    assert!(
        (elliptic_f(phi + Complex::new(core::f64::consts::PI, 0.0), m).unwrap()
            - elliptic_f(phi, m).unwrap()
            - elliptic_k(m).unwrap() * 2.0)
            .norm()
            < 1e-12
    );
    assert!((elliptic_pi(m, m).unwrap() - elliptic_e(m).unwrap() / (one - m)).norm() < 1e-12);
    let upper = elliptic_k(Complex::new(2.0, 0.0)).unwrap();
    let lower = elliptic_k(Complex::new(2.0, -0.0)).unwrap();
    assert!((upper.conj() - lower).norm() < 1e-14);
    assert!(upper.im > 0.0);
    assert!(elliptic_pi(Complex::new(2.0, 0.0), Complex::new(2.0, -0.0)).is_err());
    let identity = SolveIdentity::new(
        ProblemToken::new(502),
        scioxide_core::algorithm_id!("complex-elliptic", "test", 1),
        0,
        0,
    );
    let options = EllipticOptions::default();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let r = elliptic_pi_scoped(Complex::new(2.0, 0.5), m, options, &mut context).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(
        r,
        elliptic_pi_with_options(Complex::new(2.0, 0.5), m, options).unwrap()
    );
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        r.counts().iterations(),
        context.counts().nested().iterations()
    );
    // The parent limit is scope-local; the options budget bounds the child epoch.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let r = elliptic_pi_scoped(
        Complex::new(2.0, 0.5),
        m,
        options.with_iteration_budget(Budget::ONE),
        &mut context,
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
    assert_eq!(context.counts().nested().iterations(), 1);
    assert_eq!(
        jacobi_sn_scoped(phi, m, options, &mut context)
            .unwrap()
            .try_value()
            .unwrap(),
        jacobi_sn(phi, m).unwrap()
    );
}

#[test]
fn tiny_complement_and_exact_second_kind_limit() {
    let h = 1e-100_f64;
    let m = Complex::new(1.0, h);
    let k = elliptic_k(m).unwrap();
    let e = elliptic_e(m).unwrap();
    let logarithm = 4.0_f64.ln() - 0.5 * h.ln();
    assert!((k.re / logarithm - 1.0).abs() < 1e-14);
    assert!((k.im - core::f64::consts::FRAC_PI_4).abs() < 1e-14);
    assert!((e.im / (-0.5 * h * (logarithm - 0.5)) - 1.0).abs() < 1e-14);
    let e = elliptic_e_inc(
        Complex::new(core::f64::consts::FRAC_PI_2, 0.0),
        Complex::new(1.0, 0.0),
    )
    .unwrap();
    assert!((e - Complex::new(1.0, 0.0)).norm() < 1e-14);
}

#[test]
fn ordinary_pi_identities_conjugation_and_plain_projection() {
    let zero = Complex::new(0.0_f64, 0.0);
    let one = Complex::new(1.0, 0.0);
    for m in [Complex::new(0.375, 0.0625), Complex::new(0.5, -0.25)] {
        for n in [zero, m, Complex::new(0.25, -0.125)] {
            let plain = elliptic_pi(n, m).unwrap();
            let report = elliptic_pi_with_options(n, m, EllipticOptions::default()).unwrap();
            let detailed = report.try_value().unwrap();
            assert_eq!(plain.re.to_bits(), detailed.re.to_bits());
            assert_eq!(plain.im.to_bits(), detailed.im.to_bits());
            assert!(report.derivative().is_none());
            assert!((plain.conj() - elliptic_pi(n.conj(), m.conj()).unwrap()).norm() < 2e-13);
            let expected = if n == zero {
                elliptic_k(m).unwrap()
            } else if n == m {
                elliptic_e(m).unwrap() / (one - m)
            } else {
                carlson_rf(zero, one - m, one).unwrap()
                    + n * carlson_rj(zero, one - m, one, one - n).unwrap() / 3.0
            };
            assert!((plain - expected).norm() < 2e-13);
            let expected = Complex::new(core::f64::consts::FRAC_PI_2, 0.0) / (one - n).sqrt();
            assert!((elliptic_pi(n, zero).unwrap() - expected).norm() < 2e-14);
        }
    }
}
