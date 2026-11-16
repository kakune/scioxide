//! Independent identities, exact growth and representation/resource boundaries.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::*;
use scioxide_special::*;

#[test]
fn complex_products_preserve_independent_component_ranges() {
    for order in [1, 2] {
        let a = Complex64::new(f64::MAX, f64::from_bits(1));
        let report = pochhammer_with_options(
            a,
            Complex::new(order as _, 0.0),
            CombinatorialOptions::default(),
        )
        .unwrap();
        let value = report.scaled().unwrap().value();
        let imaginary = value.imaginary().round().unwrap();
        let expected = if order == 1 {
            f64::from_bits(1)
        } else {
            f64::from_bits((973_u64 << 52) | ((1_u64 << 52) - 1))
        };
        assert_eq!(imaginary.to_bits(), expected.to_bits());
        assert!(
            !report
                .logarithmic()
                .unwrap()
                .value()
                .scaled_phase()
                .imaginary()
                .is_zero()
        );
        if order == 1 {
            assert_eq!(value.try_value().unwrap(), a);
        }

        let a = Complex32::new(f32::MAX, f32::from_bits(1));
        let report = pochhammer_with_options(
            a,
            Complex::new(order as _, 0.0),
            CombinatorialOptions::default(),
        )
        .unwrap();
        let value = report.scaled().unwrap().value();
        let expected = if order == 1 {
            f32::from_bits(1)
        } else {
            f32::from_bits((106_u32 << 23) | ((1_u32 << 23) - 1))
        };
        assert_eq!(
            value.imaginary().round().unwrap().to_bits(),
            expected.to_bits()
        );
        assert!(
            !report
                .logarithmic()
                .unwrap()
                .value()
                .scaled_phase()
                .imaginary()
                .is_zero()
        );
        if order == 1 {
            assert_eq!(value.try_value().unwrap(), a);
        }
    }
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:.17e} != {b:.17e}"
    );
}
#[test]
fn factorial_variants_and_sequence_conventions() {
    assert_eq!(
        generalized_binomial_with_options(1.5_f64, 0, CombinatorialOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Exact
    );
    assert_eq!(factorial::<f64>(0).unwrap(), 1.0);
    assert_eq!(double_factorial::<f64>(9).unwrap(), 945.0);
    assert_eq!(multifactorial::<f32>(10, 3).unwrap(), 280.0);
    assert_eq!(pochhammer(3.0_f64, 4.0).unwrap(), 360.0);
    assert_eq!(pochhammer(-3.0_f64, 5.0).unwrap(), 0.0);
    assert_eq!(pochhammer(-3.0_f64, 5.0).unwrap(), 0.0);
    assert_eq!(pochhammer(-3.0_f64, 2.0).unwrap(), 6.0);
    close(pochhammer(3.0_f64, -2.0).unwrap(), 0.5, 1e-14);
    assert_eq!(bernoulli::<f32>(1).unwrap(), -0.5);
    assert_eq!(bernoulli::<f64>(3).unwrap(), 0.0);
    assert_eq!(euler::<f64>(2).unwrap(), -1.0);
    assert_eq!(euler::<f32>(8).unwrap(), 1385.0);
    assert_eq!(checked_bernoulli(12).unwrap(), (-691, 2730));
    assert_eq!(checked_bernoulli(usize::MAX).unwrap(), (0, 1));
    assert_eq!(checked_euler(usize::MAX).unwrap(), 0);
    assert!(checked_bernoulli(36).is_err());
    assert!(checked_euler(24).is_err());
    assert!(multifactorial::<f64>(1, 0).is_err());
    assert!(pochhammer(1.0_f64, -1.0).is_err());
    assert!(gamma(0.0_f64).is_err());
    assert!(pochhammer(f64::NAN, 0.0).is_err());
    close(
        gamma(1.5_f64).unwrap(),
        0.5 * core::f64::consts::PI.sqrt(),
        1e-14,
    );
}
#[test]
fn recurrence_and_generating_identities_in_both_precisions() {
    for n in 0..=60 {
        close(
            double_factorial::<f64>(n + 2).unwrap(),
            (n + 2) as f64 * double_factorial::<f64>(n).unwrap(),
            2e-14,
        );
        for step in 1..=5 {
            close(
                multifactorial::<f64>(n + step, step).unwrap(),
                (n + step) as f64 * multifactorial::<f64>(n, step).unwrap(),
                2e-14,
            );
        }
    }
    for n in 1..=14 {
        let mut b = 0.0;
        let mut e = 0.0;
        for k in 0..n {
            b += checked_binomial(n + 1, k).unwrap() as f64 * bernoulli::<f64>(k).unwrap();
        }
        close(b, -((n + 1) as f64) * bernoulli::<f64>(n).unwrap(), 2e-12);
        if n.is_multiple_of(2) {
            for k in (0..=n).step_by(2) {
                e += checked_binomial(n, k).unwrap() as f64 * euler::<f64>(k).unwrap();
            }
            close(e, 0.0, 1e-12);
        }
        close(
            bernoulli::<f32>(n).unwrap() as f64,
            bernoulli::<f64>(n).unwrap(),
            1e-7,
        );
        close(
            euler::<f32>(n).unwrap() as f64,
            euler::<f64>(n).unwrap(),
            1e-7,
        );
    }
    for n in 0..=15 {
        for k in 0..=n + 1 {
            close(
                stirling_first_kind::<f64>(n, k).unwrap(),
                checked_stirling_first_kind(n, k).unwrap() as f64,
                1e-14,
            );
            close(
                stirling_second_kind::<f64>(n, k).unwrap(),
                checked_stirling_second_kind(n, k).unwrap() as f64,
                1e-14,
            );
            close(
                stirling_second_kind::<f32>(n, k).unwrap() as f64,
                stirling_second_kind::<f64>(n, k).unwrap(),
                1e-7,
            );
        }
    }
}
#[test]
fn scaled_outputs_survive_ordinary_overflow() {
    let o = CombinatorialOptions::default();
    for r in [
        double_factorial_with_options::<f64>(1000, o).unwrap(),
        multifactorial_with_options(1000, 3, o).unwrap(),
        bernoulli_with_options(1000, o).unwrap(),
        euler_with_options(1000, o).unwrap(),
        factorial_with_options(1000, CombinatorialOptions::default()).unwrap(),
        binomial_with_options(2000, 1000, CombinatorialOptions::default()).unwrap(),
    ] {
        assert!(r.termination().is_success(), "{r:?}");
        assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
        assert!(r.scaled().is_some());
        assert!(
            r.logarithmic()
                .unwrap()
                .value()
                .log_magnitude()
                .unwrap()
                .is_finite()
        );
        assert!(r.try_value().is_err());
    }
    assert!(ln_binomial::<f64>(2000, 1000).unwrap().is_finite());
    assert!(ln_double_factorial::<f32>(1000).unwrap().is_finite());
    let s =
        stirling_second_kind_with_options::<f32>(150, 75, CombinatorialOptions::default()).unwrap();
    assert_eq!(s.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(s.scaled().is_some());
}
#[test]
fn pochhammer_small_shifts_and_complex_continuation() {
    for x in [0.0, 1e-300, 1e-18, -1e-18, 1e-8, -1e-8] {
        close(
            pochhammer_relative(3.25_f64, x).unwrap(),
            digamma(3.25_f64).unwrap(),
            2e-8,
        );
    }
    for a in [-2.25_f64, -0.25, 0.25, 3.25, 1e6] {
        close(
            pochhammer_relative(a, 1e-18).unwrap(),
            digamma(a).unwrap(),
            1e-13,
        );
        close(
            pochhammer(a, 3.0).unwrap(),
            a * (a + 1.0) * (a + 2.0),
            1e-13,
        );
    }
    let a = Complex::new(2.0_f64, 0.5);
    let x = Complex::new(0.25, 0.125);
    let p = pochhammer(a, x).unwrap();
    let q = pochhammer(a.conj(), x.conj()).unwrap();
    close(p.re, q.re, 1e-13);
    close(p.im, -q.im, 1e-13);
    let product = pochhammer(a, Complex::new(3.0, 0.0)).unwrap();
    let reference = a * (a + Complex::new(1.0, 0.0)) * (a + Complex::new(2.0, 0.0));
    close(product.re, reference.re, 1e-14);
    close(product.im, reference.im, 1e-14);
    assert_eq!(generalized_binomial(-3.0_f32, 3).unwrap(), -10.0);
    close(generalized_binomial(4.5_f64, 3).unwrap(), 6.5625, 1e-14);
}
#[test]
fn budgets_are_explicit() {
    let one = CombinatorialOptions::default().with_iteration_budget(Budget::ONE);
    let r = double_factorial_with_options::<f64>(20, one).unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    let _ = publication::exhausted_counts(
        &stirling_second_kind_with_options::<f64>(usize::MAX, usize::MAX - 1, one).unwrap_err(),
    );
    let budget = CombinatorialOptions::new(Budget::try_from(10000).unwrap());
    assert!(
        bernoulli_with_options::<f64>(40, budget)
            .unwrap()
            .termination()
            .is_success()
    );
}

#[test]
fn bounded_exact_sequences_match_generating_function_coefficients() {
    fn choose(n: usize, k: usize) -> i128 {
        (1..=k).fold(1, |v, i| v * (n - k + i) as i128 / i as i128)
    }
    fn gcd(mut a: i128, mut b: i128) -> i128 {
        a = a.abs();
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    fn add((a, b): (i128, i128), (c, d): (i128, i128)) -> (i128, i128) {
        let common = gcd(b, d);
        let p = a * (d / common) + c * (b / common);
        let q = b * (d / common);
        let g = gcd(p, q);
        (p / g, q / g)
    }
    let mut bernoulli = vec![(1_i128, 1_i128)];
    let mut euler = vec![1_i128];
    for n in 1..=38 {
        let (p, q) = bernoulli
            .iter()
            .enumerate()
            .fold((0, 1), |sum, (k, &(p, q))| {
                add(sum, (choose(n + 1, k) * p, q))
            });
        let q = q * (n + 1) as i128;
        let g = gcd(p, q);
        let expected = (-p / g, q / g);
        bernoulli.push(expected);
        if let (Ok(p), Ok(q)) = (isize::try_from(expected.0), usize::try_from(expected.1)) {
            assert_eq!(checked_bernoulli(n).unwrap(), (p, q));
        } else {
            assert!(checked_bernoulli(n).is_err());
        }
        if n <= 22 {
            let expected = if n.is_multiple_of(2) {
                -euler
                    .iter()
                    .enumerate()
                    .step_by(2)
                    .map(|(k, &e)| choose(n, k) * e)
                    .sum::<i128>()
            } else {
                0
            };
            euler.push(expected);
            if let Ok(expected) = isize::try_from(expected) {
                assert_eq!(checked_euler(n).unwrap(), expected);
            } else {
                assert!(checked_euler(n).is_err());
            }
        }
    }
}
#[test]
fn logarithmic_surfaces_share_work_limits_and_zero_conventions() {
    assert_eq!(ln_multifactorial::<f64>(0, 3).unwrap(), 0.0);
    assert_eq!(ln_binomial::<f32>(5, 0).unwrap(), 0.0);
    assert_eq!(pochhammer_relative(1.0_f64, 1.0).unwrap(), 0.0);
    let one = CombinatorialOptions::default().with_iteration_budget(Budget::ONE);
    let _ = publication::exhausted_counts(
        &ln_double_factorial_with_options::<f64>(100, one).unwrap_err(),
    );
    let _ =
        publication::exhausted_counts(&ln_binomial_with_options::<f64>(256, 128, one).unwrap_err());
    assert!(ln_multifactorial_with_options::<f64>(0, 0, one).is_err());
}

fn pole_neighborhood<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = CombinatorialOptions::default();
    let mut count = 0;
    for line in include_str!("data/combinatorial_poles.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let kind = fields[1];
        if fields[0] != width || !matches!(kind, "7" | "8") {
            continue;
        }
        let decode = |i: usize| {
            let bits = u64::from_str_radix(fields[i], 16).unwrap();
            let value = if width == "64" {
                f64::from_bits(bits)
            } else {
                f64::from(f32::from_bits(bits as u32))
            };
            S::try_from_f64_exact(value).unwrap()
        };
        let a = Complex::new(decode(4), decode(5));
        if a.re > S::ZERO || a.im == S::ZERO {
            continue;
        }
        let x = Complex::new(decode(6), decode(7));
        let expected = Complex::new(decode(8), decode(9));
        let report = if kind == "7" {
            pochhammer_with_options(a, x, options)
        } else {
            pochhammer_relative_with_options(a, x, options)
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        assert!(report.termination().is_success(), "{line}: {report:?}");
        let value = report.value().unwrap();
        let actual = *value.value();
        let error = (actual.re.into_f64() - expected.re.into_f64())
            .hypot(actual.im.into_f64() - expected.im.into_f64());
        let half_spacing = |v: S| (v.abs() - v.abs().next_after(S::ZERO)).into_f64() * 0.5;
        let rounding = half_spacing(expected.re).hypot(half_spacing(expected.im));
        assert!(
            error <= value.absolute_error().into_f64() + rounding,
            "{line}: error={error:e}, report={report:?}"
        );
        let norm = expected.re.into_f64().hypot(expected.im.into_f64());
        assert!(
            error <= (128.0 * f64::EPSILON + S::EPSILON.into_f64()) * norm,
            "{line}: error={error:e}, norm={norm:e}"
        );
        let logarithm = report.logarithmic().unwrap();
        let actual_log = logarithm.value().log_magnitude().unwrap().into_f64();
        let expected_log = decode(10);
        assert!(
            (actual_log - expected_log.into_f64()).abs()
                <= logarithm.absolute_error().into_f64() + half_spacing(expected_log),
            "{line}"
        );
        count += 1;
    }
    assert!(count >= 4000, "missing pole fixtures: {count}");
}

#[test]
fn pochhammer_complex_negative_integer_poles_f64() {
    pole_neighborhood::<f64>("64");
}

#[test]
fn pochhammer_complex_negative_integer_poles_f32() {
    pole_neighborhood::<f32>("32");
}
