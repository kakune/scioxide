//! Tests against independent 100-digit reference values.
#![allow(clippy::unwrap_used)]
use super::*;
use crate::{LogSurvival, Multinomial, SeedableRng};
use scioxide_core::NumericScalar;
fn observed<S: RealScalar + NumericScalar<Real = S>>(law: &str, c: bool) -> f64 {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let matrix =
        |r, n, x: &[f64]| Matrix::from_vec(r, n, x.iter().map(|&x| s(x)).collect()).unwrap();
    let u = matrix(2, 2, &[2.0, 0.25, 0.25, 1.0]);
    let col = matrix(2, 2, &[1.0, 0.2, 0.2, 3.0]);
    let mean = matrix(2, 2, &[0.2, -0.3, 0.4, 1.0]);
    let x = matrix(
        2,
        2,
        if c {
            &[20.0, -15.0, 12.0, 30.0]
        } else {
            &[0.5, -1.0, 0.75, 1.5]
        },
    );
    let point = [
        s(if c { 20.0 } else { 0.5 }),
        s(if c { -15.0 } else { -1.0 }),
    ];
    let df = s(if c { 0.75 } else { 5.5 });
    let alpha = if c {
        [s(0.01), s(1000.0), s(5.0)]
    } else {
        [s(0.125), s(2.5), s(7.0)]
    };
    let k = if c { [1, 499, 500] } else { [2, 3, 5] };
    match law {
        "multivariate_normal" => crate::MultivariateNormal::new(&[s(0.2), s(-0.3)], &u)
            .unwrap()
            .log_pdf(point.as_slice())
            .unwrap(),
        "multivariate_t" => MultivariateStudentT::with_options(
            &[s(0.2), s(-0.3)],
            &u,
            df,
            Default::default(),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pdf(&point)
        .unwrap(),
        "matrix_normal" => MatrixNormal::with_options(
            &mean,
            &u,
            &col,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pdf(x.view())
        .unwrap(),
        "matrix_t" => MatrixT::with_options(
            &mean,
            &u,
            &col,
            df,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pdf(x.view())
        .unwrap(),
        "wishart" | "invwishart" => {
            let x = matrix(
                2,
                2,
                if c {
                    &[0.01, 0.0, 0.0, 100.0]
                } else {
                    &[1.0, 0.1, 0.1, 2.0]
                },
            );
            let df = s(if c { 40.0 } else { 3.5 });
            if law == "wishart" {
                Wishart::with_options(
                    df,
                    &u,
                    crate::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap()
                .log_pdf(x.view())
                .unwrap()
            } else {
                InverseWishart::with_options(
                    df,
                    &u,
                    crate::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap()
                .log_pdf(x.view())
                .unwrap()
            }
        }
        "dirichlet" => Dirichlet::with_options(
            &alpha,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pdf(&if c {
            [s(0.001), s(0.875), s(0.124)]
        } else {
            [s(0.125), s(0.25), s(0.625)]
        })
        .unwrap(),
        "dirichlet_multinomial" => DirichletMultinomial::with_options(
            if c { 1000 } else { 10 },
            &alpha,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pmf(&k)
        .unwrap(),
        "multinomial" => Multinomial::with_options(
            if c { 1000 } else { 10 },
            &[s(0.2), s(0.3), s(0.5)],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pmf(&k)
        .unwrap(),
        "multivariate_hypergeom" => MultivariateHypergeometric::<S>::with_options(
            if c { &[100, 200, 300] } else { &[10, 20, 30] },
            if c { 80 } else { 8 },
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pmf(if c { &[1, 39, 40] } else { &[1, 3, 4] })
        .unwrap(),
        "random_table" => RandomTable::<S>::with_options(
            if c { &[100, 200] } else { &[3, 4] },
            if c { &[80, 120, 100] } else { &[2, 3, 2] },
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pmf(if c {
            &[1, 50, 49, 79, 70, 51]
        } else {
            &[1, 1, 1, 1, 2, 1]
        })
        .unwrap(),
        "gaussian_log_upper" => {
            crate::MultivariateNormal::new(&[s(0.0); 2], &Matrix::identity(2).unwrap())
                .unwrap()
                .log_sf(&[s(if c { 40.0 } else { 8.0 }); 2])
                .unwrap()
        }
        "joint_normal" => {
            let g =
                crate::MultivariateNormal::new(&[s(0.0); 2], &matrix(2, 2, &[1.0, 0.5, 0.5, 1.0]))
                    .unwrap();
            let plan = g
                .prepare_probability(
                    MultivariateProbabilityOptions::default()
                        .with_points_per_replicate(65536)
                        .with_tolerance(scioxide_core::Tolerance::new(0.0, 1e-4).unwrap()),
                    &crate::Philox4x64::from_seed(542),
                )
                .unwrap();
            let report = plan
                .rectangle(
                    &[f64::NEG_INFINITY; 2],
                    if c { &[-8.0, -9.0] } else { &[1.0, -0.5] },
                )
                .unwrap();
            assert_eq!(
                report.termination,
                MultivariateProbabilityTermination::Converged,
                "case {c}: {report:?}"
            );
            s(report.log_probability)
        }
        "normal_inverse_gamma" => NormalInverseGamma::with_options(
            s(0.2),
            s(2.5),
            s(if c { 0.25 } else { 3.5 }),
            s(if c { 0.01 } else { 4.0 }),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap()
        .log_pdf((
            s(if c { 100.0 } else { 1.0 }),
            s(if c { 0.001 } else { 2.0 }),
        ))
        .unwrap(),
        _ => unreachable!(),
    }
    .into_f64()
}
#[test]
fn independent_defining_formula_fixtures_both_widths() {
    let mut count = 0;
    for row in include_str!("../../../tests/data/multivariate.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let r = row.split('\t').collect::<Vec<_>>();
        let expected = r[3].parse::<f64>().unwrap();
        let single = r[1] == "f32";
        let actual = if single {
            observed::<f32>(r[0], r[2] == "1")
        } else {
            observed::<f64>(r[0], r[2] == "1")
        };
        let tolerance = if r[0] == "joint_normal" {
            5e-6
        } else if single {
            2e-6
        } else {
            5e-12
        };
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{} {} case {}: {actual} != {expected}",
            r[0],
            r[1],
            r[2]
        );
        assert_eq!(expected.to_bits(), u64::from_str_radix(r[5], 16).unwrap());
        count += 1;
    }
    assert_eq!(count, 56);
}
