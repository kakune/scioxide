#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_random::*;
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * b.abs().max(1.0), "{a} != {b}");
}

fn finite_ks_reference<S: scioxide_core::RealScalar>(width: &str) {
    let options =
        LawOptions::default().with_work_budget(scioxide_core::Budget::try_from(1 << 30).unwrap());
    for line in include_str!("../data/finite_ks.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let columns: Vec<_> = line.split('\t').collect();
        if columns[0] != width {
            continue;
        }
        let n = columns[1].parse().unwrap();
        let x = S::try_from_f64_lossy(columns[2].parse().unwrap()).unwrap();
        let law = KolmogorovSmirnovTwoSided::with_options(n, options).unwrap();
        let mut scratch = law.workspace(x).unwrap();
        for (column, actual) in [(3, law.log_cdf(x).unwrap()), (4, law.log_sf(x).unwrap())] {
            let expected = columns[column].parse::<f64>().unwrap();
            let tolerance = 2e-10 + 4.0 * S::EPSILON.into_f64() * expected.abs();
            assert!(
                (actual.into_f64() - expected).abs() <= tolerance,
                "{line}: {actual:?} vs {expected}"
            );
            let report = law.evaluate_tail(x, column == 4, &mut scratch).unwrap();
            close(
                report.log_probability.into_f64(),
                actual.into_f64(),
                tolerance,
            );
            if width == "64" {
                assert!(
                    (report.log_probability.into_f64().exp() - expected.exp()).abs()
                        <= report.absolute_error.into_f64(),
                    "{line}: {report:?}"
                );
            }
        }
        if columns[5] != "-" {
            let expected = columns[5].parse::<f64>().unwrap();
            let actual = law.log_pdf(x).unwrap().into_f64();
            let tolerance = 2e-10 + 4.0 * S::EPSILON.into_f64() * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{line}: density {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn finite_ks_independent_f64() {
    finite_ks_reference::<f64>("64");
}
#[test]
fn finite_ks_independent_f32() {
    finite_ks_reference::<f32>("32");
}

#[test]
fn finite_ks_integer_knots_and_support() {
    let options =
        LawOptions::default().with_work_budget(scioxide_core::Budget::try_from(1 << 30).unwrap());
    for n in [1, 31, 512, 513, 4096] {
        let law = KolmogorovSmirnovTwoSided::<f64>::with_options(n, options).unwrap();
        close(law.cdf(0.5 / n as f64).unwrap(), 0.0, 1e-15);
        close(law.sf(1.0).unwrap(), 0.0, 1e-15);
        if n == 1 {
            continue;
        }
        let knot = ((n as f64).sqrt() * 0.6).ceil() / n as f64;
        let coordinates = [knot.next_down(), knot, knot.next_up()];
        let mut previous = 0.0;
        for x in coordinates {
            let lower = law.cdf(x).unwrap();
            let upper = law.sf(x).unwrap();
            assert!(lower >= previous - 2e-11);
            close(lower + upper, 1.0, 2e-14);
            assert!(law.pdf(x).unwrap() > 0.0);
            previous = lower;
        }
    }
}
#[test]
fn central_and_mixture_identities() {
    let f = FisherF::with_options(
        2.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [1e-50, 0.1, 1.0, 10.0, 1e50] {
        close(f.log_cdf(x).unwrap(), x.ln() - (1.0 + x).ln(), 1e-12);
        close(f.log_sf(x).unwrap(), -(1.0 + x).ln(), 1e-12);
        close(f.log_pdf(x).unwrap(), -2.0 * (1.0 + x).ln(), 1e-12);
    }
    let n = NoncentralChiSquared::with_options(
        1.0_f64,
        4.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let normal = Normal::new(2.0, 1.0).unwrap();
    for x in [0.001_f64, 0.2, 1.0, 5.0, 30.0] {
        close(
            n.cdf(x).unwrap(),
            normal.cdf(x.sqrt()).unwrap() - normal.cdf(-x.sqrt()).unwrap(),
            1e-8,
        );
    }
    let n = NoncentralStudentT::with_options(
        5.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    close(
        n.cdf(0.0).unwrap(),
        Normal::new(0.0, 1.0).unwrap().cdf(-2.0).unwrap(),
        1e-12,
    );
    let r = Rice::with_options(
        0.0_f64,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [0.1_f64, 1.0, 10.0] {
        close(r.log_sf(x).unwrap(), -x * x / 2.0, 1e-10);
    }
}
#[test]
fn ks_exact_small_samples() {
    let one = KolmogorovSmirnovOneSided::<f64>::with_options(
        1,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let two = KolmogorovSmirnovTwoSided::<f64>::with_options(
        1,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [0.1, 0.3, 0.7, 0.9] {
        close(one.cdf(x).unwrap(), x, 1e-12);
        close(one.pdf(x).unwrap(), 1.0, 1e-12);
        close(two.cdf(x).unwrap(), (2.0 * x - 1.0).max(0.0), 1e-12);
    }
    let two = KolmogorovSmirnovTwoSided::<f64>::with_options(
        5,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    close(two.sf(0.5).unwrap(), 0.112, 1e-12);
    close(two.cdf(0.3).unwrap(), 0.336, 1e-12);
    close(
        KolmogorovSmirnovOneSided::<f64>::with_options(
            1,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .mean()
        .unwrap(),
        0.5,
        1e-10,
    );
}
#[test]
fn conditional_integrals() {
    let t = NoncentralStudentT::with_options(
        5.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [-3.0, 1.0, 3.0] {
        let p = t.cdf(x).unwrap();
        let s = t.sf(x).unwrap();
        close(p + s, 1.0, 1e-8);
        assert!(t.pdf(x).unwrap() > 0.0);
    }
    let r = StudentizedRange::with_options(
        3.0_f64,
        10.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let p = r.cdf(3.0).unwrap();
    let s = r.sf(3.0).unwrap();
    close(p + s, 1.0, 2e-8);
    close(p, 0.8650165848104374, 2e-8);
}
#[test]
fn f32_surface() {
    close(
        FisherF::with_options(
            2.0_f32,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .cdf(1.0)
        .unwrap() as f64,
        0.5,
        1e-6,
    );
    close(
        NoncentralF::with_options(
            2.0_f32,
            2.0,
            0.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .cdf(1.0)
        .unwrap() as f64,
        0.5,
        1e-6,
    );
    close(
        KolmogorovSmirnovTwoSided::<f32>::with_options(
            5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .cdf(0.3)
        .unwrap() as f64,
        0.336,
        1e-5,
    );
}

#[test]
fn range_work_and_moments() {
    let o = LawOptions::default()
        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        .with_tolerance(scioxide_core::Tolerance::new(1.25e-10, 1.25e-10).unwrap());
    let r = StudentizedRange::with_options(3.0_f64, 15.0, o).unwrap();
    for x in [0.1, 1.0, 2.0, 3.0, 4.0, 8.0] {
        let report = r.evaluate_tail(x, true);
        assert!(report.is_ok());
    }
}

fn fixture_law<S: scioxide_core::RealScalar>(
    name: &str,
    a: S,
    b: S,
    c: S,
) -> Box<dyn ContinuousLaw<S>> {
    match name {
        "f" => Box::new(
            FisherF::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "ncx2" => Box::new(
            NoncentralChiSquared::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "ncf" => Box::new(
            NoncentralF::with_options(
                a,
                b,
                c,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "nct" => Box::new(
            NoncentralStudentT::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "rice" => Box::new(
            Rice::with_options(
                a,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "ksone" => Box::new(
            KolmogorovSmirnovOneSided::<S>::with_options(
                a.into_f64() as usize,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "kstwo" => Box::new(
            KolmogorovSmirnovTwoSided::<S>::with_options(
                a.into_f64() as usize,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "kstwobign" => Box::new(
            Kolmogorov::<S>::with_options(
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "studentized_range" => Box::new(
            StudentizedRange::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        _ => panic!("unknown fixture"),
    }
}
fn reference<S: scioxide_core::RealScalar>() {
    let tolerance = if S::EPSILON.into_f64() > 1e-10 {
        3e-5
    } else {
        8e-8
    };
    for line in include_str!("../data/test_statistic_laws.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<_> = line.split('\t').collect();
        let cv = |i: usize| {
            let x = v[i].parse::<f64>().unwrap();
            if x == f64::INFINITY {
                S::INFINITY
            } else {
                S::try_from_f64_lossy(x).unwrap()
            }
        };
        let Ok(x) = S::try_from_f64_lossy(v[4].parse::<f64>().unwrap()) else {
            continue;
        };
        if !x.is_finite() || x == S::ZERO {
            continue;
        }
        let law = fixture_law(v[0], cv(1), cv(2), cv(3));
        for (i, actual) in [law.log_pdf(x), law.log_cdf(x), law.log_sf(x)]
            .into_iter()
            .enumerate()
        {
            let actual = actual
                .unwrap_or_else(|e| panic!("{line} column {i}: {e}"))
                .into_f64();
            let expected = v[i + 5].parse::<f64>().unwrap();
            assert!(
                (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                "{} x={} column {i}: {actual} != {expected}",
                v[0],
                v[4]
            );
        }
    }
}
#[test]
fn independent_f64() {
    reference::<f64>();
}
#[test]
fn independent_f32() {
    reference::<f32>();
}

#[test]
fn oracle_quantiles_and_extreme_log_inversion() {
    for line in include_str!("../data/test_statistic_laws.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<_> = line.split('\t').collect();
        let value = |i: usize| v[i].parse::<f64>().unwrap();
        let law = fixture_law(v[0], value(1), value(2), value(3));
        let upper = value(7) < value(6);
        let lp = if upper { value(7) } else { value(6) };
        let default = InversionOptions::default();
        let root = default
            .root_options()
            .clone()
            .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 1e-10).unwrap());
        let result = invert_log_probability_with_options(
            &*law,
            lp,
            upper,
            &InversionOptions::default()
                .with_root_options(root)
                .with_work_budget(scioxide_core::Budget::try_from(1_048_576).unwrap()),
        )
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        assert!(result.root().is_success(), "{line}");
        close(result.value() / value(4), 1.0, 2e-6);
    }
}
fn goodness<S: scioxide_core::RealScalar, D: Distribution<S> + Quantile<S> + Cdf<S>>(law: D) {
    let mut rng = SplitMix64::from_seed(540);
    statistics::continuous(core::any::type_name::<D>(), &law, || {
        law.sample(&mut rng).unwrap()
    });
}
fn seeded<S: scioxide_core::RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    goodness(
        FisherF::with_options(
            s(5.0),
            s(11.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        NoncentralChiSquared::with_options(
            s(5.0),
            s(100.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        NoncentralF::with_options(
            s(5.0),
            s(11.0),
            s(3.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        NoncentralStudentT::with_options(
            s(5.0),
            s(2.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        Rice::with_options(
            s(3.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        KolmogorovSmirnovOneSided::<S>::with_options(
            5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        KolmogorovSmirnovTwoSided::<S>::with_options(
            5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        Kolmogorov::<S>::with_options(
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness(
        StudentizedRange::with_options(
            s(3.0),
            s(10.0),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}
#[test]
fn seeded_f64() {
    seeded::<f64>();
}
#[test]
fn seeded_f32() {
    seeded::<f32>();
}

#[test]
fn normalization_and_moments() {
    let laws: [Box<dyn ContinuousLaw<f64>>; 9] = [
        Box::new(
            FisherF::with_options(
                5.0,
                11.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralF::with_options(
                5.0,
                11.0,
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralChiSquared::with_options(
                5.0,
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            NoncentralStudentT::with_options(
                5.0,
                2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            Rice::with_options(
                3.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            KolmogorovSmirnovOneSided::with_options(
                5,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            KolmogorovSmirnovTwoSided::with_options(
                5,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            Kolmogorov::with_options(
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        Box::new(
            StudentizedRange::with_options(
                3.0,
                f64::INFINITY,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
    ];
    for (i, law) in laws.iter().enumerate() {
        let n = if i == 5 || i == 6 { 10240 } else { 2048 };
        let sum = (0..n)
            .map(|j| {
                let t = (j as f64 + 0.5) / n as f64;
                let (x, jac) = if i == 3 {
                    let z = core::f64::consts::PI * (t - 0.5);
                    (z.tan(), core::f64::consts::PI / z.cos().powi(2))
                } else if i == 5 || i == 6 {
                    (t, 1.0)
                } else {
                    (t / (1.0 - t), 1.0 / (1.0 - t).powi(2))
                };
                law.pdf(x).unwrap() * jac
            })
            .sum::<f64>()
            / n as f64;
        assert!((sum - 1.0).abs() < 3e-5, "law {i} normalized to {sum}");
    }
    close(
        FisherF::with_options(
            5.0_f64,
            11.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .mean()
        .unwrap(),
        11.0 / 9.0,
        1e-12,
    );
    close(
        NoncentralChiSquared::with_options(
            5.0_f64,
            3.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .variance()
        .unwrap(),
        22.0,
        1e-12,
    );
    close(
        Rice::with_options(
            0.0_f64,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .mean()
        .unwrap(),
        (core::f64::consts::PI / 2.0).sqrt(),
        1e-12,
    );
    close(
        Rice::with_options(
            0.0_f64,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .variance()
        .unwrap(),
        2.0 - core::f64::consts::PI / 2.0,
        1e-12,
    );
    close(
        KolmogorovSmirnovTwoSided::<f64>::with_options(
            1,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .mean()
        .unwrap(),
        0.75,
        1e-8,
    );
    close(
        KolmogorovSmirnovTwoSided::<f64>::with_options(
            1,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .variance()
        .unwrap(),
        1.0 / 48.0,
        1e-8,
    );
    close(
        StudentizedRange::with_options(
            3.0_f64,
            f64::INFINITY,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .mean()
        .unwrap(),
        3.0 / core::f64::consts::PI.sqrt(),
        1e-7,
    );
}
#[test]
fn domains_limits_and_budgets() {
    use scioxide_core::Budget;
    assert!(
        FisherF::with_options(
            0.0_f64,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        NoncentralF::with_options(
            1.0_f64,
            1.0,
            -1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        NoncentralChiSquared::with_options(
            0.0_f64,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        NoncentralStudentT::with_options(
            1.0_f64,
            f64::NAN,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        Rice::with_options(
            -1.0_f64,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        KolmogorovSmirnovOneSided::<f64>::with_options(
            0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        StudentizedRange::with_options(
            1.0_f64,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    let error = FisherF::with_options(
        5.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap()
    .mean()
    .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    let error = NoncentralStudentT::with_options(
        2.0_f64,
        1.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap()
    .variance()
    .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    assert!(
        StudentizedRange::with_options(
            3.0_f64,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .mean()
        .is_err()
    );
    let o = LawOptions::default().with_work_budget(Budget::ONE);
    assert!(
        NoncentralChiSquared::with_options(5.0_f64, 100.0, o)
            .unwrap()
            .cdf(100.0)
            .is_err()
    );
    let ks = KolmogorovSmirnovTwoSided::<f64>::with_options(
        20,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut empty = KolmogorovWorkspace::new();
    assert_eq!(
        ks.evaluate_tail(0.15, false, &mut empty)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert!(ks.evaluate_tail(f64::NAN, false, &mut empty).is_err());
    // The law's byte limit bounds both reservation and retained storage.
    let limited = KolmogorovSmirnovTwoSided::<f64>::with_options(
        20,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
            .with_max_bytes(1),
    )
    .unwrap();
    let requirements = limited.workspace_requirements(0.15).unwrap();
    assert_eq!(requirements.max_bytes(), 1);
    assert_eq!(
        KolmogorovWorkspace::new()
            .try_reserve(&requirements)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        limited.log_cdf(0.15).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let mut reserved = ks.workspace(0.15).unwrap();
    assert_eq!(
        limited
            .evaluate_tail(0.15, false, &mut reserved)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let f = FisherF::with_options(
        1.0_f64,
        5.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(f.pdf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(f.quantile(0.0).unwrap(), 0.0);
    assert_eq!(f.isf(0.0).unwrap(), f64::INFINITY);
    assert!(f.cdf(f64::NAN).is_err());
    assert!(f.quantile(-0.1).is_err());
    for x in [0.1_f64, 1.0, 3.0] {
        close(
            NoncentralStudentT::with_options(
                5.0_f64,
                -2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap()
            .cdf(-x)
            .unwrap(),
            NoncentralStudentT::with_options(
                5.0_f64,
                2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap()
            .sf(x)
            .unwrap(),
            1e-8,
        );
    }
    assert!(
        KolmogorovSmirnovTwoSided::<f64>::with_options(
            100,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .log_sf(0.4)
        .unwrap()
        .is_finite()
    );
}

#[test]
fn independent_moments_and_large_degree_limits() {
    for row in include_str!("../data/test_statistic_moments.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<_> = row.split('\t').collect();
        let n = |i: usize| v[i].parse::<f64>().unwrap();
        let (mean, variance) = match v[0] {
            "nct" => {
                let law = NoncentralStudentT::with_options(
                    n(1),
                    n(2),
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap();
                (law.mean().unwrap(), law.variance().unwrap())
            }
            "rice" => {
                let law = Rice::with_options(
                    n(1),
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap();
                (law.mean().unwrap(), law.variance().unwrap())
            }
            "studentized_range" => {
                let law = StudentizedRange::with_options(
                    n(1),
                    n(2),
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap();
                (law.mean().unwrap(), law.variance().unwrap())
            }
            _ => panic!("unknown moment fixture"),
        };
        close(mean, n(3), 2e-8);
        close(variance, n(4), 2e-8);
    }
    let finite = NoncentralStudentT::with_options(
        1e8_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let limit = Normal::new(2.0, 1.0).unwrap();
    close(finite.cdf(2.0).unwrap(), limit.cdf(2.0).unwrap(), 1e-7);
}

#[test]
fn noninteger_range_and_small_degrees() {
    let range = StudentizedRange::with_options(
        1.5_f64,
        2.5,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [0.01, 1.0, 8.0] {
        close(range.cdf(x).unwrap() + range.sf(x).unwrap(), 1.0, 1e-7);
        assert!(range.pdf(x).unwrap() > 0.0);
    }
    let t = NoncentralStudentT::with_options(
        0.1_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    close(t.cdf(1.0).unwrap() + t.sf(1.0).unwrap(), 1.0, 1e-7);
}

#[test]
fn noncentral_t_extreme_power_tail_remains_logarithmic() {
    let law = NoncentralStudentT::with_options(
        5.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let anchor = law.log_sf(1e100).unwrap();
    for x in [1e200_f64, 1e308] {
        close(
            law.log_sf(x).unwrap() - anchor,
            -5.0 * (x.ln() - 1e100_f64.ln()),
            1e-9,
        );
    }
}

#[test]
fn mixture_index_limits_return_errors() {
    let limit = (usize::MAX as u128).min(1_u128 << 52) as f64;
    let law = NoncentralChiSquared::with_options(
        5.0_f64,
        2.0 * (limit + 1.0),
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert!(law.cdf(2.0 * limit).is_err());
}

#[path = "../support/statistics.rs"]
mod statistics;

/// Force every interior bin-boundary probability through the fractional-range
/// sampling branch. Generic inversion and uniform decoding have separate GoF
/// coverage; this deterministic defining-integral check avoids thousands of
/// repeated expensive inversions and also checks default numerical accuracy.
#[test]
fn fractional_range_inverse_mapping_both_widths() {
    fractional_range::<f32>();
    fractional_range::<f64>();
}
struct ProbabilityWord(u64);
impl BitGenerator for ProbabilityWord {
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok((self.0 >> 32) as u32)
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        Ok(self.0)
    }
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        Err(scioxide_core::SciOxideError::not_implemented(
            "word fixture does not fill bytes",
        ))
    }
}
impl<S: scioxide_core::RealScalar> Rng<S> for ProbabilityWord {
    fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
        Ok(decode_u64(self.0))
    }
}
fn fractional_range<S: scioxide_core::RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let law = StudentizedRange::with_options(
        s(3.5),
        S::INFINITY,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for i in 1_u32..32 {
        // This word decodes exactly to i/32 in both scalar widths.
        let mut rng = ProbabilityWord(u64::from(i) << 59);
        let q = law.sample(&mut rng).unwrap();
        assert!(q.is_finite());
        // Independently integrate k*phi(x)*(Phi(x+q)-Phi(x))^(k-1). The
        // production evaluator uses adaptive log-tail integrals; fixed Simpson
        // panels resolve this smooth integrand on [-12,12], with omitted mass
        // <1e-31. The numerical fixture allowance also covers quadrature error.
        let f = |x: f64| {
            let interval = scioxide_special::normal_cdf(x + q.into_f64()).unwrap()
                - scioxide_special::normal_cdf(x).unwrap();
            3.5 * (-0.5 * x * x).exp() / core::f64::consts::TAU.sqrt() * interval.max(0.0).powf(2.5)
        };
        let h = 24.0 / 4096.0;
        let sum: f64 = (1..4096)
            .map(|j| f(-12.0 + f64::from(j) * h) * if j % 2 == 0 { 2.0 } else { 4.0 })
            .sum();
        let cdf = h * (f(-12.0) + f(12.0) + sum) / 3.0;
        assert!((f64::from(i) / 32.0 - cdf).abs() < statistics::reference_error::<S>());
    }
}
