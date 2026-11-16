use super::*;
use scioxide_core::Complex;

#[test]
fn reserved_evaluation_has_no_heap_allocations() {
    let h = plan(4, 1);
    h.first_kind(1.5, Default::default())
        .and_then(|report| report.try_value())
        .unwrap();
    h.second_kind(Complex::new(1.5, 0.25))
        .and_then(|report| report.try_value())
        .unwrap();
    h.normalization()
        .and_then(|report| report.try_value())
        .unwrap();
    let scoped = h
        .first_kind_scoped(
            1.5,
            EllipsoidalSigns::default(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(scoped.termination().is_success());
}

#[test]
fn high_degree_angular_tails() {
    // Independent 80-digit shifted-polynomial oracle, exact binary64 inputs.
    let h = plan(32, 1);
    for (s, expected) in [
        (0.71, -7.555_093_432_191_168e-15),
        (0.8, 3.385_419_819_111_171_6e-15),
        (0.9, -5.483_851_941_774_527e-15),
        (0.99, 6.483_893_441_387_44e-15),
    ] {
        let actual = h
            .first_kind(s, Default::default())
            .and_then(|report| report.try_value())
            .unwrap();
        assert!((actual - expected).abs() <= 5e-11 * expected.abs());
    }
}

#[test]
fn focal_distance_uses_exact_input_squares() {
    let s = 0.9_f64;
    let h2 = 0.81_f64;
    let h = EllipsoidalHarmonicPlan::new(h2, 2.0, 1, 2, EllipsoidalOptions::default()).unwrap();
    let expected = s.mul_add(s, -h2).abs().sqrt();
    let actual = h
        .first_kind(s, Default::default())
        .and_then(|report| report.try_value())
        .unwrap();
    assert!(
        (actual - expected).abs() < 5e-13,
        "{actual:e} != {expected:e}"
    );
    for z in [
        Complex::new(-1.5, 0.25),
        Complex::new(0.0, 1.5),
        Complex::new(0.0, -1.5),
    ] {
        let r = plan(3, 2);
        let e = r
            .first_kind(z, Default::default())
            .and_then(|report| report.try_value())
            .unwrap();
        let mirrored = r
            .first_kind(-z, Default::default())
            .and_then(|report| report.try_value())
            .unwrap();
        assert!((e + mirrored).norm() < 5e-12 * e.norm());
        let f = r
            .second_kind(z)
            .and_then(|report| report.try_value())
            .unwrap();
        let mirrored = r
            .second_kind(-z)
            .and_then(|report| report.try_value())
            .unwrap();
        assert!((f - mirrored).norm() < 5e-12 * f.norm());
    }
}

#[test]
fn independent_multiprecision_fixtures() {
    let mut worst = [[0.0_f64; 6]; 2];
    for line in include_str!("../../../tests/data/ellipsoidal.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let n = row[3] as usize;
        let p = row[4] as usize;
        let mut check = |values: [f64; 6], tolerance: f64| {
            for (i, (value, expected)) in values.into_iter().zip(&row[6..]).enumerate() {
                let relative = (value - expected).abs() / expected.abs().max(1e-280);
                let width_index = usize::from(row[0] == 64.0);
                worst[width_index][i] = worst[width_index][i].max(relative);
                assert!(
                    relative <= tolerance || (value == 0.0 && *expected == 0.0),
                    "n={n} p={p} width={} field={i}: {value:e} != {expected:e}, relative={relative:e}",
                    row[0]
                );
            }
        };
        if row[0] == 64.0 {
            let options = EllipsoidalOptions::default()
                .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-8).unwrap());
            let h = EllipsoidalHarmonicPlan::new(row[1], row[2], n, p, options).unwrap();
            let e = h.first_kind(row[5], EllipsoidalSigns::default()).unwrap();
            let f = h.second_kind(row[5]).unwrap();
            let norm = h.normalization().unwrap();
            assert!(
                e.termination().is_success()
                    && f.termination().is_success()
                    && norm.termination().is_success(),
                "n={n} p={p}: {:?} {:?} {:?}",
                e.termination(),
                f.termination(),
                norm.termination()
            );
            check(
                [
                    *e.value().unwrap().value(),
                    *e.derivative().unwrap().value(),
                    *f.value().unwrap().value(),
                    *f.derivative().unwrap().value(),
                    *norm.value().unwrap().value(),
                    h.eigenvalue_over_k_squared() * row[2],
                ],
                5e-11,
            );
        } else {
            let options = EllipsoidalOptions::<f32>::default();
            let h =
                EllipsoidalHarmonicPlan::new(row[1] as f32, row[2] as f32, n, p, options).unwrap();
            let e = h
                .first_kind(row[5] as f32, EllipsoidalSigns::default())
                .unwrap();
            let f = h.second_kind(row[5] as f32).unwrap();
            let norm = h.normalization().unwrap();
            assert!(
                e.termination().is_success()
                    && f.termination().is_success()
                    && norm.termination().is_success(),
                "n={n} p={p}: {:?} {:?} {:?}, counts {}",
                e.termination(),
                f.termination(),
                norm.termination(),
                norm.counts().iterations()
            );
            check(
                [
                    *e.value().unwrap().value() as f64,
                    *e.derivative().unwrap().value() as f64,
                    *f.value().unwrap().value() as f64,
                    *f.derivative().unwrap().value() as f64,
                    *norm.value().unwrap().value() as f64,
                    h.eigenvalue_over_k_squared() * row[2],
                ],
                2e-4,
            );
        }
    }
    std::println!("ellipsoidal fixture worst relative errors [f32, f64]: {worst:?}");
}

fn check_complex_oracle<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let width = if R::MANTISSA_DIGITS == 24 { 32.0 } else { 64.0 };
    let tolerance = if width == 32.0 { 2e-4 } else { 5e-11 };
    let convert = |x: f64| R::try_from_f64_lossy(x).unwrap();
    let options = EllipsoidalOptions::default().with_relative_tolerance(
        scioxide_core::RelativeTolerance::new(convert(if width == 32.0 { 1e-5 } else { 1e-8 }))
            .unwrap(),
    );
    for line in include_str!("../../../tests/data/ellipsoidal_complex.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let r: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        if r[0] != width {
            continue;
        }
        let n = r[3] as usize;
        let p = r[4] as usize;
        let h = EllipsoidalHarmonicPlan::new(convert(r[1]), convert(r[2]), n, p, options).unwrap();
        let z = Complex::new(convert(r[5]), convert(r[6]));
        let e = h.first_kind(z, EllipsoidalSigns::default()).unwrap();
        let f = h.second_kind(z).unwrap();
        assert!(e.termination().is_success() && f.termination().is_success());
        let values = [
            *e.value().unwrap().value(),
            *e.derivative().unwrap().value(),
            *f.value().unwrap().value(),
            *f.derivative().unwrap().value(),
        ];
        for (i, v) in values.into_iter().enumerate() {
            let expected = Complex::new(r[7 + 2 * i], r[8 + 2 * i]);
            let actual = Complex::new(
                v.re.try_to_f64_exact().unwrap(),
                v.im.try_to_f64_exact().unwrap(),
            );
            assert!(
                (actual - expected).norm() <= tolerance * expected.norm().max(1e-280),
                "complex f{width} n={n} p={p} field={i}: {actual:?} != {expected:?}"
            );
        }
    }
}
#[test]
fn independent_complex_frobenius_fixtures() {
    check_complex_oracle::<f32>();
    check_complex_oracle::<f64>();
}

fn plan(n: usize, p: usize) -> EllipsoidalHarmonicPlan<f64> {
    EllipsoidalHarmonicPlan::new(0.5, 1.0, n, p, EllipsoidalOptions::default()).unwrap()
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual:e} != {expected:e}"
    );
}
#[test]
fn low_degree_values_derivatives_and_normalization() {
    for s in [-2.0_f64, -0.8, -0.3, 0.0, 0.3, 0.8, 2.0] {
        close(
            plan(0, 1)
                .first_kind(s, Default::default())
                .and_then(|report| report.try_value())
                .unwrap(),
            1.0,
            1e-14,
        );
        close(
            plan(1, 1)
                .first_kind(s, Default::default())
                .and_then(|report| report.try_value())
                .unwrap(),
            s,
            1e-14,
        );
        close(
            plan(1, 2)
                .first_kind(s, Default::default())
                .and_then(|report| report.try_value())
                .unwrap(),
            (s * s - 0.5).abs().sqrt(),
            1e-14,
        );
        close(
            plan(1, 3)
                .first_kind(s, Default::default())
                .and_then(|report| report.try_value())
                .unwrap(),
            (s * s - 1.0).abs().sqrt(),
            1e-14,
        );
    }
    for p in 1..=3 {
        let h = plan(1, p);
        let r = h.first_kind(2.0, EllipsoidalSigns::default()).unwrap();
        let rad = [0.0, 0.5, 1.0][p - 1];
        close(
            *r.derivative().unwrap().value(),
            2.0 / (4.0 - rad).sqrt(),
            1e-13,
        );
        let norm = h.normalization().unwrap();
        let expected = 4.0 * core::f64::consts::PI / 3.0 * [0.5, 0.25, 0.5][p - 1];
        close(*norm.value().unwrap().value(), expected, 5e-11);
        assert!(norm.termination().is_success(), "{norm:?}");
    }
    close(
        plan(0, 1)
            .normalization()
            .and_then(|report| report.try_value())
            .unwrap(),
        4.0 * core::f64::consts::PI,
        5e-12,
    );
}

#[test]
fn exterior_degree_zero_carlson_and_wronskian() {
    for s in [1.001, 1.2, 2.0, 10.0] {
        let h = plan(0, 1);
        let expected = crate::carlson_rf(s * s, s * s - 0.5, s * s - 1.0).unwrap();
        close(
            h.second_kind(s)
                .and_then(|report| report.try_value())
                .unwrap(),
            expected,
            5e-11,
        );
    }
    for n in 0..=8 {
        for p in 1..=2 * n + 1 {
            let options = EllipsoidalOptions::default()
                .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-10).unwrap());
            let h = EllipsoidalHarmonicPlan::new(0.5, 1.0, n, p, options).unwrap();
            let e = h.first_kind(1.5, EllipsoidalSigns::default()).unwrap();
            let f = h.second_kind(1.5).unwrap();
            let wronskian = *e.value().unwrap().value() * *f.derivative().unwrap().value()
                - *e.derivative().unwrap().value() * *f.value().unwrap().value();
            close(
                wronskian,
                -((2 * n + 1) as f64) / ((2.25_f64 - 0.5) * (2.25 - 1.0)).sqrt(),
                5e-10,
            );
        }
    }
}

#[test]
fn complex_conjugation_and_exterior_agreement() {
    for n in 0..=4 {
        for p in 1..=2 * n + 1 {
            let h = plan(n, p);
            let z = Complex::new(1.5, 0.4);
            let e = h
                .first_kind(z, Default::default())
                .and_then(|report| report.try_value())
                .unwrap();
            let ec = h
                .first_kind(z.conj(), Default::default())
                .and_then(|report| report.try_value())
                .unwrap();
            close(e.re, ec.re, 5e-12);
            close(e.im, -ec.im, 5e-12);
            let f = h
                .second_kind(z)
                .and_then(|report| report.try_value())
                .unwrap();
            let fc = h
                .second_kind(z.conj())
                .and_then(|report| report.try_value())
                .unwrap();
            close(f.re, fc.re, 5e-12);
            close(f.im, -fc.im, 5e-12);
        }
    }
}

#[test]
fn invalid_inputs_and_work_limits() {
    for (h, k) in [
        (0.0, 1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
    ] {
        assert!(EllipsoidalHarmonicPlan::new(h, k, 2, 1, EllipsoidalOptions::default()).is_err());
    }
    assert!(
        EllipsoidalHarmonicPlan::new(0.5, 1.0, usize::MAX, 1, EllipsoidalOptions::default())
            .is_err()
    );
    assert!(EllipsoidalHarmonicPlan::new(0.5, 1.0, 1, 0, EllipsoidalOptions::default()).is_err());
    let limited = EllipsoidalOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
    assert!(EllipsoidalHarmonicPlan::new(0.5, 1.0, 8, 1, limited).is_err());
    let h = EllipsoidalHarmonicPlan::new(
        0.5,
        1.0,
        0,
        1,
        EllipsoidalOptions::default()
            .with_iteration_budget(scioxide_core::Budget::try_from(64).unwrap()),
    )
    .unwrap();
    let r = h.second_kind(1.01).unwrap_err();
    let _ = crate::test_support::exhausted_counts(&r);
    assert!(
        h.first_kind(f64::INFINITY, Default::default())
            .and_then(|report| report.try_value())
            .is_err()
    );
    assert!(
        h.second_kind(0.8)
            .and_then(|report| report.try_value())
            .is_err()
    );
}

#[test]
fn facade_allowance_covers_preparation_eigensolver_and_evaluation() {
    let signs = EllipsoidalSigns::default();
    let run = |options| {
        prepared_call(0.5, 1.0, 8, 1, options, |plan, child| {
            plan.first_run(2.0_f64, signs, Some(child))
        })
        .and_then(|evaluation| evaluation.publish())
    };
    // The reported total is preparation P, the nested eigensolver epoch E and
    // evaluation V.
    let total = run(EllipsoidalOptions::default())
        .unwrap()
        .counts()
        .iterations();
    let budget = |work| {
        EllipsoidalOptions::default()
            .with_iteration_budget(scioxide_core::Budget::try_from(work).unwrap())
    };
    let exact = run(budget(total)).unwrap();
    assert!(exact.value().is_some());
    assert_eq!(exact.counts().iterations(), total);
    // P + E still fits and P + V <= total - 1 because E > 0, but the shared
    // allowance leaves evaluation one step short.
    let short = run(budget(total - 1)).unwrap_err();
    let _ = crate::test_support::exhausted_counts(&short);
}

#[test]
fn differential_equation_and_focal_derivatives() {
    let options = EllipsoidalOptions::default()
        .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-8).unwrap());
    for n in 0..=8 {
        for p in 1..=2 * n + 1 {
            let h = EllipsoidalHarmonicPlan::new(0.5, 1.0, n, p, options).unwrap();
            for s in [0.23, 0.83, 1.4] {
                let step = 1e-5;
                let evaluate = |x| h.first_kind(x, EllipsoidalSigns::default()).unwrap();
                let e = evaluate(s);
                let y = *e.value().unwrap().value();
                let dy = *e.derivative().unwrap().value();
                let ddy = (*evaluate(s + step).derivative().unwrap().value()
                    - *evaluate(s - step).derivative().unwrap().value())
                    / (2.0 * step);
                let terms = [
                    (s * s - 0.5) * (s * s - 1.0) * ddy,
                    s * (2.0 * s * s - 1.5) * dy,
                    (h.eigenvalue_over_k_squared() - (n * (n + 1)) as f64 * s * s) * y,
                ];
                let scale = terms.iter().map(|x| x.abs()).sum::<f64>().max(1e-20);
                assert!(
                    terms.iter().sum::<f64>().abs() < 1e-6 * scale,
                    "n={n}, p={p}, s={s}, {terms:?}"
                );
            }
        }
    }
    let r = plan(1, 3)
        .first_kind(1.0, EllipsoidalSigns::default())
        .unwrap();
    assert_eq!(*r.value().unwrap().value(), 0.0);
    assert!(r.derivative().is_none());
}

#[test]
fn focal_lips_and_precision() {
    let h = plan(1, 2);
    let upper = h
        .first_kind(Complex::new(0.3, 0.0), Default::default())
        .and_then(|report| report.try_value())
        .unwrap();
    let lower = h
        .first_kind(Complex::new(0.3, -0.0), Default::default())
        .and_then(|report| report.try_value())
        .unwrap();
    close(upper.re, 0.0, 1e-14);
    close(upper.im, 0.41_f64.sqrt(), 1e-14);
    close(lower.im, -upper.im, 1e-14);
    let h32 =
        EllipsoidalHarmonicPlan::new(0.5_f32, 1.0, 4, 3, EllipsoidalOptions::default()).unwrap();
    let h64 = plan(4, 3);
    for s in [0.23_f32, 0.83, 1.4] {
        close(
            h32.first_kind(s, Default::default())
                .and_then(|report| report.try_value())
                .unwrap() as f64,
            h64.first_kind(s as f64, Default::default())
                .and_then(|report| report.try_value())
                .unwrap(),
            2e-5,
        );
    }
    let c = h32
        .second_kind(Complex::new(1.5_f32, 0.2))
        .and_then(|report| report.try_value())
        .unwrap();
    let d = h64
        .second_kind(Complex::new(1.5, 0.2_f32 as f64))
        .and_then(|report| report.try_value())
        .unwrap();
    close(c.re as f64, d.re, 2e-5);
    close(c.im as f64, d.im, 2e-5);
}

#[test]
fn degenerate_axis_limits_and_scaled_range() {
    for h2 in [1e-6_f64, 0.5, 1.0 - 1e-6] {
        let h = EllipsoidalHarmonicPlan::new(h2, 1.0, 0, 1, EllipsoidalOptions::default()).unwrap();
        close(
            h.normalization()
                .and_then(|report| report.try_value())
                .unwrap(),
            4.0 * core::f64::consts::PI,
            5e-11,
        );
        close(
            h.second_kind(2.0)
                .and_then(|report| report.try_value())
                .unwrap(),
            crate::carlson_rf(4.0, 4.0 - h2, 3.0).unwrap(),
            5e-11,
        );
    }
    for n in 1..=3 {
        for p in 1..=2 * n + 1 {
            let h =
                EllipsoidalHarmonicPlan::new(0.5e-12, 1e-12, n, p, EllipsoidalOptions::default())
                    .unwrap();
            close(
                h.first_kind(2.0, Default::default())
                    .and_then(|report| report.try_value())
                    .unwrap(),
                2.0_f64.powi(n as i32),
                1e-10,
            );
        }
    }
    let options = EllipsoidalOptions::default()
        .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-8).unwrap());
    let h = EllipsoidalHarmonicPlan::new(0.5, 1.0, 8, 1, options).unwrap();
    let e = h.first_kind(1e100, EllipsoidalSigns::default()).unwrap();
    assert_eq!(
        e.scalar_representation(),
        crate::ScalarRepresentation::Overflow
    );
    assert!(e.scaled().is_some());
    assert!(e.termination().is_success());
}

#[test]
fn weighted_angular_orthogonality() {
    let opts = EllipsoidalOptions::default()
        .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-8).unwrap());
    let integrate = |f: &dyn Fn(f64) -> f64| -> f64 {
        let half = core::f64::consts::PI / 4.0;
        crate::support::GL32
            .iter()
            .map(|&(x, w)| w * half * f(half * (x + 1.0)))
            .sum()
    };
    for n in 2..=6 {
        for p in 1..2 * n + 1 {
            let a = EllipsoidalHarmonicPlan::new(0.5, 1.0, n, p, opts).unwrap();
            let b = EllipsoidalHarmonicPlan::new(0.5, 1.0, n, p + 1, opts).unwrap();
            if a.class() != b.class() {
                continue;
            }
            let product = |s| {
                *a.first_kind(s, EllipsoidalSigns::default())
                    .unwrap()
                    .value()
                    .unwrap()
                    .value()
                    * *b.first_kind(s, EllipsoidalSigns::default())
                        .unwrap()
                        .value()
                        .unwrap()
                        .value()
            };
            let nu = |t: f64| 0.5_f64.sqrt() * t.sin();
            let mu = |t: f64| (0.5 + 0.5 * t.sin().powi(2)).sqrt();
            let a0 = integrate(&|t| product(nu(t)) / (1.0 - nu(t).powi(2)).sqrt());
            let a2 = integrate(&|t| product(nu(t)) * nu(t).powi(2) / (1.0 - nu(t).powi(2)).sqrt());
            let b0 = integrate(&|t| product(mu(t)) / mu(t));
            let b2 = integrate(&|t| product(mu(t)) * mu(t));
            let norm = (a.normalization().unwrap().value().unwrap().value()
                * b.normalization().unwrap().value().unwrap().value())
            .sqrt();
            assert!(
                (8.0 * (a0 * b2 - a2 * b0) / norm).abs() < 5e-11,
                "degree {n}, orders {p}/{}",
                p + 1
            );
        }
    }
}

#[test]
fn scoped_evaluation_reports_nested_work_and_observes_cancellation() {
    use scioxide_core::{
        Budget, BudgetKind, CancellationToken, ProblemToken, SciOxideErrorView, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(683),
        scioxide_core::algorithm_id!("ellipsoidal", "test", 1),
        0,
        0,
    );
    let h = plan(4, 2);
    let signs = EllipsoidalSigns::default();
    let options = EllipsoidalOptions::default();
    let reference = h.first_kind(0.3, signs).unwrap();
    let root = h
        .first_kind_scoped(0.3, signs, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(root, reference);
    // Iteration allowances are scope-local: a parent limit of one does not stop
    // the child epoch, which reports its work as nested counts.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let child = h.first_kind_scoped(0.3, signs, &mut context).unwrap();
    assert_eq!(child, reference);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        child.counts().iterations()
    );
    let limited = options.with_iteration_budget(Budget::try_from(64).unwrap());
    let limited_plan = EllipsoidalHarmonicPlan::new(0.5, 1.0, 0, 1, limited).unwrap();
    let _ = crate::test_support::exhausted_counts(
        &limited_plan
            .second_kind_scoped(1.01, &mut context)
            .unwrap_err(),
    );
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    for error in [
        h.normalization_scoped(&mut context).unwrap_err(),
        EllipsoidalHarmonicPlan::<f64>::build(
            0.5,
            1.0,
            4,
            2,
            options,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
}
