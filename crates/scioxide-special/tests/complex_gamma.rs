#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{
    Budget, Complex, NumericScalar, NumericalIssue, RealScalar, SciOxideErrorView,
};
use scioxide_special::*;

#[test]
fn polygamma_plain_and_derivative_forms_share_one_value() {
    fn check<S: RealScalar>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        for n in [0, 1, 5, 20] {
            for (re, im) in [(0.75, 0.25), (-3.5, -0.125), (20.0, 1.0)] {
                let z = Complex::new(
                    S::try_from_f64_lossy(re).unwrap(),
                    S::try_from_f64_lossy(im).unwrap(),
                );
                let (plain, report) = (|| {
                    Ok::<_, scioxide_core::SciOxideError>((
                        polygamma(n, z)?,
                        polygamma_with_options(n, z, GammaOptions::default())?,
                    ))
                })()
                .unwrap();
                let expected = report.try_value().unwrap();
                assert_eq!(
                    plain.re.into_f64().to_bits(),
                    expected.re.into_f64().to_bits()
                );
                assert_eq!(
                    plain.im.into_f64().to_bits(),
                    expected.im.into_f64().to_bits()
                );
                let next = polygamma_with_options(n + 1, z, GammaOptions::default()).unwrap();
                let derivative = report.derivative().unwrap();
                let expected = next.try_value().unwrap();
                assert!(
                    (*derivative.value() - expected).norm()
                        <= derivative.absolute_error() + next.value().unwrap().absolute_error()
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn polygamma_derivative_overflow_preserves_a_finite_value() {
    for (n, z) in [
        (0, Complex::new(-1.0_f64, 1e-170)),
        (5, Complex::new(0.0, 1e-45)),
    ] {
        let report = polygamma_with_options(n, z, GammaOptions::default()).unwrap();
        assert!(report.try_value().unwrap().is_finite());
        assert_eq!(report.try_value().unwrap(), polygamma(n, z).unwrap());
        assert!(report.derivative().is_none());
    }
    for (n, z) in [
        (0, Complex::new(-1.0_f32, 1e-30)),
        (5, Complex::new(0.0, 1e-6)),
    ] {
        let report = polygamma_with_options(n, z, GammaOptions::default()).unwrap();
        assert!(report.try_value().unwrap().is_finite());
        assert_eq!(report.try_value().unwrap(), polygamma(n, z).unwrap());
        assert!(report.derivative().is_none());
    }
}

fn incomplete_report<S: RealScalar>(
    name: &str,
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
) -> Result<SpecialFunctionResult<Complex<S>>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    match name {
        "lower" => gamma_lower_with_options(a, z, options),
        "upper" => gamma_upper_with_options(a, z, options),
        "p" => gamma_p_with_options(a, z, options),
        "q" => gamma_q_with_options(a, z, options),
        _ => unreachable!(),
    }
}

#[test]
fn analytic_log_branch_and_recurrence() {
    for im in [0.0_f64, -0.0, 0.125, -0.125, 1000.0, -1000.0] {
        for re in [-20.5, -3.125, -1.5, -0.5, 0.5, 12.25] {
            let z = Complex::new(re, im);
            let a = ln_gamma(z).unwrap();
            let b = ln_gamma(Complex::new(z.re + 1.0, z.im)).unwrap();
            assert!((b - a - z.ln()).norm() < 2e-10, "{z:?}: {a:?} {b:?}");
            let conjugate = ln_gamma(z.conj()).unwrap();
            assert!((conjugate - a.conj()).norm() < 2e-10);
        }
    }
    let upper = ln_gamma(Complex::new(-3.5_f64, 0.0)).unwrap();
    let lower = ln_gamma(Complex::new(-3.5_f64, -0.0)).unwrap();
    assert!((upper.im + 4.0 * core::f64::consts::PI).abs() < 1e-13);
    assert!((lower.im - 4.0 * core::f64::consts::PI).abs() < 1e-13);
}

#[test]
fn scaled_and_budgeted() {
    let near_pole =
        ln_gamma_with_options(Complex::new(-2.0_f64, 1e-320), GammaOptions::default()).unwrap();
    assert!(near_pole.termination().is_success());
    assert!(near_pole.derivative().is_none());
    assert!(near_pole.try_value().unwrap().is_finite());
    let r = gamma_with_options(Complex::new(200.0_f64, 2.0), GammaOptions::default()).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.value().is_none());
    assert!(r.scaled().is_some() && r.logarithmic().is_some());
    let r = gamma_with_options(
        Complex::new(-2.5_f64, 1.0),
        GammaOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    {
        let outcome = &(gamma(Complex::new(-2.0_f64, 0.0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

fn complete_projection_contract<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let number = |x| S::try_from_f64_lossy(x).unwrap();
    let half = number(0.5);
    let tiny = S::ZERO.next_after(S::ONE);
    let mut points = vec![
        Complex::new(half.next_after(S::ZERO), number(0.25)),
        Complex::new(half, number(0.25)),
        Complex::new(half.next_after(S::ONE), number(0.25)),
        Complex::new(number(-2.0), tiny),
        Complex::new(number(-2.0), -tiny),
    ];
    points.extend(
        [
            (1.0, 0.0),
            (2.0, -0.0),
            (-3.5, 0.0),
            (-3.5, -0.0),
            (0.125, 0.25),
            (12.25, -3.5),
            (200.0, 2.0),
            (-20.5, 1000.0),
            (0.25, 60.0),
            (0.25, 64.0),
            (0.25, 80.0),
            (0.25, 460.0),
            (0.25, 480.0),
            (171.5, 1e-30),
            (-170.25, 1e-30),
            (-2.0, 0.0),
            (0.0, -0.0),
        ]
        .map(|(re, im)| Complex::new(number(re), number(im))),
    );
    points.extend([
        Complex::new(S::NAN, S::ONE),
        Complex::new(S::ONE, S::INFINITY),
        Complex::new(S::MAX, S::ONE),
    ]);
    for z in points {
        for logarithm in [false, true] {
            let plain = if logarithm { ln_gamma(z) } else { gamma(z) };
            let detailed = (if logarithm {
                ln_gamma_with_options(z, GammaOptions::default())
            } else {
                gamma_with_options(z, GammaOptions::default())
            })
            .and_then(SpecialFunctionResult::try_value);
            match (plain, detailed) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(
                        a.re.into_f64().to_bits(),
                        b.re.into_f64().to_bits(),
                        "{z:?}"
                    );
                    assert_eq!(
                        a.im.into_f64().to_bits(),
                        b.im.into_f64().to_bits(),
                        "{z:?}"
                    );
                }
                (Err(a), Err(b)) => {
                    assert_eq!(a.kind(), b.kind(), "{z:?}");
                    match (a.view(), b.view()) {
                        (
                            SciOxideErrorView::NumericalInstability { issue: a, .. },
                            SciOxideErrorView::NumericalInstability { issue: b, .. },
                        ) => assert_eq!(a, b),
                        (
                            SciOxideErrorView::DomainError { reason: a, .. },
                            SciOxideErrorView::DomainError { reason: b, .. },
                        ) => assert_eq!(a, b),
                        (
                            SciOxideErrorView::NonFinite { origin: a, .. },
                            SciOxideErrorView::NonFinite { origin: b, .. },
                        ) => assert_eq!(a, b),
                        _ => panic!("unexpected errors: {a:?}, {b:?}"),
                    }
                }
                values => panic!("{z:?}, logarithm={logarithm}: {values:?}"),
            }
        }
    }
}

#[test]
fn complete_plain_detailed_boundaries_and_allocations() {
    complete_projection_contract::<f32>();
    complete_projection_contract::<f64>();
    let tiny = gamma(Complex::new(0.25_f64, 460.0)).unwrap();
    assert!(tiny.re.is_subnormal() && tiny.im.is_subnormal());
    let mixed = gamma(Complex::new(171.5_f64, 1e-300)).unwrap();
    assert!(mixed.re > 1e307 && mixed.im > 0.0 && mixed.im < 1e10);
    let near_pole = Complex::new(-2.0_f64, 1e-320);
    let detailed = ln_gamma_with_options(near_pole, GammaOptions::default()).unwrap();
    assert!(detailed.derivative().is_none());
    assert_eq!(ln_gamma(near_pole).unwrap(), detailed.try_value().unwrap());
}

#[test]
fn complete_work_boundaries() {
    for (z, steps) in [
        (Complex::new(0.5_f64, 0.25), 12),
        (Complex::new(-2.5, 0.25), 13),
    ] {
        for logarithm in [false, true] {
            for budget in [steps - 1, steps, steps + 1] {
                let options = GammaOptions::default()
                    .with_iteration_budget(Budget::try_from(budget).unwrap());
                let report = if logarithm {
                    ln_gamma_with_options(z, options)
                } else {
                    gamma_with_options(z, options)
                };
                if budget < steps {
                    assert_eq!(
                        publication::exhausted_counts(&report.unwrap_err()).iterations(),
                        budget
                    );
                } else {
                    let report = report.unwrap();
                    assert_eq!(report.counts().iterations(), steps);
                    assert_eq!(report.termination(), SpecialFunctionTermination::Converged);
                    assert!(report.value().is_some() && report.derivative().is_some());
                }
            }
        }
    }
}

fn complete_derivatives<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for line in include_str!("data/complex_gamma.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] != width || fields[1] != "digamma" {
            continue;
        }
        let number = |i: usize| S::try_from_f64_lossy(fields[i].parse::<f64>().unwrap()).unwrap();
        let z = Complex::new(number(2), number(3));
        let expected = Complex::new(number(6), number(7));
        let logarithm = ln_gamma_with_options(z, GammaOptions::default()).unwrap();
        let derivative = logarithm.derivative().unwrap();
        assert!(
            (*derivative.value() - expected).norm().into_f64()
                <= derivative.absolute_error().into_f64()
                    + 4.0 * S::EPSILON.into_f64() * expected.norm().into_f64()
                    + 2e-14,
            "{line}"
        );
        let report = gamma_with_options(z, GammaOptions::default()).unwrap();
        if let (Some(value), Some(derivative)) = (report.value(), report.derivative()) {
            let expected = *value.value() * expected;
            assert!(
                (*derivative.value() - expected).norm().into_f64()
                    <= derivative.absolute_error().into_f64()
                        + (4.0 * S::EPSILON.into_f64() + 2e-14) * expected.norm().into_f64(),
                "{line}"
            );
        }
    }
}

#[test]
fn complete_derivatives_follow_independent_digamma_fixtures() {
    complete_derivatives::<f32>("32");
    complete_derivatives::<f64>("64");
}

#[test]
fn complex_incomplete_relations() {
    for a in [Complex::new(0.5_f64, 0.25), Complex::new(3.0, -0.5)] {
        for z in [Complex::new(0.5, 0.125), Complex::new(4.0, 1.0)] {
            let lower = gamma_lower(a, z).unwrap();
            let upper = gamma_upper(a, z).unwrap();
            let g = gamma(a).unwrap();
            assert!(
                (lower + upper - g).norm() < 5e-10 * g.norm().max(1.0),
                "{a:?} {z:?}"
            );
            let p = gamma_p(a, z).unwrap();
            let q = gamma_q(a, z).unwrap();
            assert!((p + q - Complex::new(1.0, 0.0)).norm() < 5e-10);
        }
    }
    let z = Complex::new(1.0_f64, 0.25);
    assert_eq!(
        gamma_p(Complex::new(-2.0, 0.0), z).unwrap(),
        Complex::new(1.0, 0.0)
    );
    assert_eq!(
        gamma_q(Complex::new(-2.0, 0.0), z).unwrap(),
        Complex::new(0.0, 0.0)
    );
    assert!(gamma_upper(Complex::new(-2.0, 0.0), z).is_ok());
}

#[test]
fn beta_and_f32() {
    let a = Complex::new(2.5_f64, 0.75);
    let b = Complex::new(1.25_f64, -0.25);
    let actual = beta(a, b).unwrap();
    let expected = gamma(a).unwrap() * gamma(b).unwrap() / gamma(a + b).unwrap();
    assert!((actual - expected).norm() < 2e-12);
    let g = gamma(Complex::new(5.0_f32, 0.0)).unwrap();
    assert!((g.re - 24.0).abs() < 2e-5 && g.im.abs() < 2e-5);
    let l = ln_gamma(Complex::new(-3.5_f32, -0.0)).unwrap();
    assert!((l.im - 4.0 * core::f32::consts::PI).abs() < 2e-5);
}

fn oracle<S: scioxide_core::RealScalar>(width: &str)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    for line in include_str!("data/complex_gamma.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let number = |i: usize| S::try_from_f64_lossy(f[i].parse::<f64>().unwrap()).unwrap();
        let a = Complex::new(number(2), number(3));
        let z = Complex::new(number(4), number(5));
        let expected = Complex::new(number(6), number(7));
        let actual = match f[1] {
            "loggamma" => ln_gamma(a).unwrap(),
            "gamma" => {
                let r = gamma_with_options(a, GammaOptions::default()).unwrap();
                assert!(r.termination().is_success(), "{line}: {r:?}");
                let log = r.logarithmic().unwrap().value();
                let phase = log.scaled_phase().round().unwrap();
                // Compare phases on the circle, including negative real Gamma.
                let p = Complex::new(expected.im.cos(), expected.im.sin());
                assert!(
                    (phase - p).norm()
                        < S::try_from_f64_lossy(if width == "32" { 2e-5 } else { 2e-12 }).unwrap(),
                    "{line}"
                );
                Complex::new(log.log_magnitude().unwrap(), expected.im)
            }
            "digamma" => digamma(a).unwrap(),
            "polygamma" => polygamma(2, a).unwrap(),
            "zeta" => riemann_zeta(a).unwrap(),
            "eta" => dirichlet_eta(a).unwrap(),
            "hurwitz" => hurwitz_zeta(a, z).unwrap(),
            "beta_inc" => beta_inc(
                a,
                z,
                Complex::new(
                    S::try_from_f64_lossy(0.25).unwrap(),
                    S::try_from_f64_lossy(0.125).unwrap(),
                ),
            )
            .unwrap(),
            "beta" => beta(a, z).unwrap(),
            "lower" | "upper" | "p" | "q" => {
                let report = incomplete_report(f[1], a, z, GammaOptions::default())
                    .unwrap_or_else(|error| panic!("{line}: {error:?}"));
                assert!(report.termination().is_success(), "{line}: {report:?}");
                let estimate = report.value().unwrap();
                let error = (*estimate.value() - expected).norm().into_f64();
                let rounding = 2.0 * S::EPSILON.into_f64() * expected.norm().into_f64();
                assert!(
                    error <= estimate.absolute_error().into_f64() + rounding,
                    "{line}: error={error:e}, bound={:e}, value={:?}",
                    estimate.absolute_error().into_f64(),
                    *estimate.value()
                );
                *estimate.value()
            }
            _ => unreachable!(),
        };
        let incomplete = matches!(f[1], "lower" | "upper" | "p" | "q" | "beta_inc");
        let (rel, abs) = if matches!(f[1], "zeta" | "eta" | "hurwitz") {
            if width == "64" {
                (5e-9_f64, 5e-12_f64)
            } else {
                (1e-3, 2e-5)
            }
        } else {
            match (width, incomplete) {
                ("64", false) => (2e-12_f64, 2e-14_f64),
                ("64", true) => (2e-10, 2e-13),
                ("32", false) => (2e-5, 2e-6),
                _ => (5e-4, 2e-6),
            }
        };

        if matches!(f[1], "lower" | "upper" | "p" | "q") {
            // A rounded complex phase can move a normwise rounding error into
            // an exactly zero component, especially on an integral cut lip.
            // Check the complex norm as well as the reported bound above.
            let error = (actual - expected).norm().into_f64();
            let limit = abs.max(rel * expected.norm().into_f64());
            assert!(
                error <= limit,
                "{line}: {actual:?} error {error} limit {limit}"
            );
            continue;
        }
        for (v, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
            let error = (v - e).abs().try_to_f64_exact().unwrap();
            let limit = abs.max(rel * e.abs().try_to_f64_exact().unwrap());
            assert!(
                error <= limit,
                "{line}: {actual:?} error {error} limit {limit}"
            );
        }
    }
}
#[test]
fn independent_f64() {
    oracle::<f64>("64");
}
#[test]
fn independent_f32() {
    oracle::<f32>("32");
}

/// SplitMix64, with the issue number as the seed; no platform entropy.
struct Samples(u64);
impl Samples {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut bits = self.0;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let unit = ((bits ^ (bits >> 31)) >> 11) as f64 * 2.0_f64.powi(-53);
        lo + (hi - lo) * unit
    }
    fn log(&mut self, lo: f64, hi: f64) -> f64 {
        self.uniform(lo.ln(), hi.ln()).exp()
    }
    fn point(&mut self, index: usize) -> (Complex<f64>, Complex<f64>) {
        let (a, radius, phase) = if index < 1200 {
            let re = self.uniform(0.1, 20.0);
            let im = if (400..800).contains(&index) {
                self.uniform(1.0, 5.0) * if index.is_multiple_of(2) { 1.0 } else { -1.0 }
            } else {
                self.uniform(-0.1, 0.1)
            };
            let radius = if index < 800 {
                self.log(1e-3, 0.1)
            } else {
                self.log(1.0, 50.0)
            };
            (
                Complex::new(re, im),
                radius,
                self.uniform(-core::f64::consts::PI, core::f64::consts::PI),
            )
        } else {
            let a = Complex::new(
                if index.is_multiple_of(4) {
                    self.uniform(-20.0, -0.1)
                } else {
                    self.log(1e-4, 200.0)
                },
                self.uniform(-20.0, 20.0),
            );
            let radius = if index.is_multiple_of(3) {
                a.norm() * self.uniform(0.8, 1.2)
            } else {
                self.log(1e-6, 200.0)
            };
            (
                a,
                radius,
                self.uniform(-core::f64::consts::PI, core::f64::consts::PI),
            )
        };
        (a, Complex::new(radius * phase.cos(), radius * phase.sin()))
    }
}

fn sweep<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut samples = Samples(1150);
    let mut failures = Vec::new();
    for index in 0..2048 {
        let (a, z) = samples.point(index);
        let narrow = |c: Complex<f64>| {
            Complex::new(
                S::try_from_f64_lossy(c.re).unwrap(),
                S::try_from_f64_lossy(c.im).unwrap(),
            )
        };
        let (a, z) = (narrow(a), narrow(z));
        for (name, result) in [
            ("lower", gamma_lower(a, z)),
            ("upper", gamma_upper(a, z)),
            ("p", gamma_p(a, z)),
            ("q", gamma_q(a, z)),
        ] {
            let valid = match &result {
                Ok(value) => value.is_finite(),
                Err(error) => matches!(
                    error.view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow | NumericalIssue::Underflow,
                        ..
                    }
                ),
            };
            if !valid {
                let coordinates = [a.re, a.im, z.re, z.im].map(RealScalar::into_f64);
                failures.push(format!("{name} seed=1150 index={index} precision={} args={coordinates:?} bits={:x?}: {result:?}", core::any::type_name::<S>(), coordinates.map(f64::to_bits)));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn documented_domain_f64() {
    sweep::<f64>();
}

#[test]
fn documented_domain_f32() {
    sweep::<f32>();
}

#[test]
fn reproducer_derivatives_budgets_and_allocations() {
    let a = Complex::new(2.7881610989887076_f64, 0.03470671342846998);
    let z = Complex::new(0.02842616704752339, -0.005068492501103518);
    for (a, z) in [
        (a, z),
        (Complex::new(20.0, 0.01), Complex::new(20.0, 0.1)),
        (Complex::new(3.0, -0.1), Complex::new(80.0, 5.0)),
        (Complex::new(2.5, 0.1), Complex::new(-0.1, -0.0)),
    ] {
        let density = ((a - Complex::new(1.0, 0.0)) * z.ln() - z).exp();
        let complete = gamma(a).unwrap();
        for name in ["lower", "upper", "p", "q"] {
            let report = incomplete_report(name, a, z, GammaOptions::default()).unwrap();
            assert!(
                report.termination().is_success(),
                "{name} {a:?} {z:?}: {report:?}"
            );
            let expected = density
                * Complex::new(
                    if matches!(name, "upper" | "q") {
                        -1.0
                    } else {
                        1.0
                    },
                    0.0,
                )
                / if matches!(name, "p" | "q") {
                    complete
                } else {
                    Complex::new(1.0, 0.0)
                };
            let derivative = report.derivative().unwrap();
            assert!(
                (*derivative.value() - expected).norm()
                    <= derivative.absolute_error() + 16.0 * f64::EPSILON * expected.norm(),
                "{name} {a:?} {z:?}: {derivative:?} expected {expected:?}"
            );
            let plain = match name {
                "lower" => gamma_lower(a, z),
                "upper" => gamma_upper(a, z),
                "p" => gamma_p(a, z),
                "q" => gamma_q(a, z),
                _ => unreachable!(),
            }
            .unwrap();
            let value = report.try_value().unwrap();
            assert_eq!(
                [plain.re.to_bits(), plain.im.to_bits()],
                [value.re.to_bits(), value.im.to_bits()]
            );
            let limited = incomplete_report(
                name,
                a,
                z,
                GammaOptions::default().with_iteration_budget(Budget::ONE),
            )
            .unwrap_err();
            let limited_counts = publication::exhausted_counts(&limited);
            assert_eq!(limited_counts.iterations(), 1);
        }
    }
}

#[test]
fn incomplete_scopes_charge_the_shared_budget() {
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let a = Complex::new(2.7881610989887076_f64, 0.03470671342846998);
    let z = Complex::new(0.02842616704752339, -0.005068492501103518);
    for name in ["lower", "upper", "p", "q"] {
        let options = GammaOptions::default();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(1150),
                    scioxide_core::algorithm_id!("gamma", "test", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let actual = match name {
            "lower" => gamma_lower_scoped(a, z, options, &mut context),
            "upper" => gamma_upper_scoped(a, z, options, &mut context),
            "p" => gamma_p_scoped(a, z, options, &mut context),
            "q" => gamma_q_scoped(a, z, options, &mut context),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(actual, incomplete_report(name, a, z, options).unwrap());
        assert_eq!(
            context.counts().nested().iterations(),
            actual.counts().iterations()
        );
        assert_eq!(context.counts().own().iterations(), 0);
    }
}

#[test]
fn incomplete_endpoints_cuts_and_range() {
    let zero = Complex::new(0.0_f64, 0.0);
    let one = Complex::new(1.0_f64, 0.0);
    let a = Complex::new(2.5, 0.01);
    assert_eq!(gamma_p(a, zero).unwrap(), zero);
    assert_eq!(gamma_lower(a, zero).unwrap(), zero);
    assert_eq!(gamma_q(a, zero).unwrap(), one);
    assert_eq!(gamma_upper(a, zero).unwrap(), gamma(a).unwrap());
    for name in ["lower", "upper", "p", "q"] {
        for z in [
            Complex::new(-0.1, 0.0),
            Complex::new(-0.1, -0.0),
            Complex::new(20.0, 0.1),
        ] {
            let first = incomplete_report(name, a, z, GammaOptions::default()).unwrap();
            let conjugate =
                incomplete_report(name, a.conj(), z.conj(), GammaOptions::default()).unwrap();
            assert!(
                (first.try_value().unwrap().conj() - conjugate.try_value().unwrap()).norm()
                    <= first.value().unwrap().absolute_error()
                        + conjugate.value().unwrap().absolute_error()
            );
        }
        for (a, z) in [
            (Complex::new(f64::INFINITY, 0.0), one),
            (one, Complex::new(0.0, f64::NAN)),
        ] {
            assert!(matches!(
                incomplete_report(name, a, z, GammaOptions::default())
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::NonFinite { .. }
            ));
        }
        assert!(matches!(
            incomplete_report(name, Complex::new(-0.5, 1.0), zero, GammaOptions::default())
                .unwrap_err()
                .view(),
            SciOxideErrorView::DomainError { .. }
        ));
    }
    let overflow = gamma_upper_with_options(
        Complex::new(200.0, 0.1),
        Complex::new(0.01, 0.001),
        GammaOptions::default(),
    )
    .unwrap();
    assert!(overflow.termination().is_success());
    assert_eq!(
        overflow.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(overflow.scaled().is_some() && overflow.logarithmic().is_some());
}
