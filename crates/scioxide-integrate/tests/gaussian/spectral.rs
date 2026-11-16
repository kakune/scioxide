//! Independent Gaussian spectral data, endpoint exactness, and failure contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use scioxide_core::{Budget, Interval, RealScalar};
use scioxide_integrate::{
    FixedRule, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
    ProductRule,
};
use scioxide_linalg::LinalgExecutionOptions;
use scioxide_poly::OrthogonalFamily;

fn canonical<S: RealScalar + Default>() -> GaussianSupport<S> {
    GaussianSupport::Interval(Interval::new(-S::ONE, S::ONE).unwrap())
}

macro_rules! width_cases {
    ($name:ident, $s:ty, $tol:expr) => {
        mod $name {
            use super::*;
            fn close(actual: $s, expected: $s) {
                assert!(
                    (actual - expected).abs() <= $tol * expected.abs().max(1.0),
                    "{actual} != {expected}"
                );
            }
            #[test]
            fn independent_nodes_and_weights() {
                let rows: Vec<_> = include_str!("../data/gaussian.csv")
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .collect();
                for group in rows.chunks_exact(6) {
                    let fields: Vec<_> = group[0].split(',').collect();
                    let alpha: $s = fields[3].parse().unwrap();
                    let beta: $s = fields[4].parse().unwrap();
                    let family = match fields[0] {
                        "legendre" => OrthogonalFamily::Legendre,
                        "jacobi" => OrthogonalFamily::Jacobi { alpha, beta },
                        "laguerre" => OrthogonalFamily::GeneralizedLaguerre { alpha },
                        "hermite_h" => OrthogonalFamily::HermiteH,
                        "hermite_he" => OrthogonalFamily::HermiteHe,
                        _ => unreachable!(),
                    };
                    let variant = match fields[1] {
                        "gauss" => GaussianRuleVariant::Gauss,
                        "left" => GaussianRuleVariant::RadauLeft,
                        "right" => GaussianRuleVariant::RadauRight,
                        "lobatto" => GaussianRuleVariant::Lobatto,
                        _ => unreachable!(),
                    };
                    let plan = GaussianRulePlan::<$s>::with_options(
                        family,
                        6,
                        variant,
                        GaussianRuleOptions::default(),
                    )
                    .unwrap();
                    for ((&node, &weight), row) in
                        plan.nodes().iter().zip(plan.weights()).zip(group)
                    {
                        let fields: Vec<_> = row.split(',').collect();
                        close(node, fields[5].parse().unwrap());
                        let expected: $s = fields[6].parse().unwrap();
                        assert!(
                            (weight / expected - 1.0).abs() < $tol * 8.0,
                            "{family:?} {variant:?}: {weight} != {expected}"
                        );
                    }
                }
            }
            #[test]
            fn endpoint_moments_and_minimum_orders() {
                let moment = |k: usize| {
                    if k.is_multiple_of(2) {
                        2.0 / (k + 1) as $s
                    } else {
                        0.0
                    }
                };
                for variant in [
                    GaussianRuleVariant::Gauss,
                    GaussianRuleVariant::RadauLeft,
                    GaussianRuleVariant::RadauRight,
                    GaussianRuleVariant::Lobatto,
                ] {
                    for n in 1..=12 {
                        if n == 1 && variant == GaussianRuleVariant::Lobatto {
                            continue;
                        }
                        for family in [
                            OrthogonalFamily::Legendre,
                            OrthogonalFamily::Jacobi {
                                alpha: 1.0,
                                beta: 2.0,
                            },
                        ] {
                            let plan = GaussianRulePlan::<$s>::with_options(
                                family,
                                n,
                                variant,
                                GaussianRuleOptions::default(),
                            )
                            .unwrap();
                            assert!(plan.nodes().windows(2).all(|pair| pair[0] < pair[1]));
                            assert!(plan.weights().iter().all(|&weight| weight > 0.0));
                            close(plan.weights().iter().sum(), plan.zeroth_moment());
                            if matches!(
                                variant,
                                GaussianRuleVariant::RadauLeft | GaussianRuleVariant::Lobatto
                            ) {
                                assert_eq!(plan.nodes()[0], -1.0);
                            }
                            if matches!(
                                variant,
                                GaussianRuleVariant::RadauRight | GaussianRuleVariant::Lobatto
                            ) {
                                assert_eq!(plan.nodes()[n - 1], 1.0);
                            }
                            for k in 0..=plan.exactness_degree() {
                                let expected = if family == OrthogonalFamily::Legendre {
                                    moment(k)
                                } else {
                                    moment(k) + moment(k + 1) - moment(k + 2) - moment(k + 3)
                                };
                                let f = |x: $s| Ok(x.powi(k as i32));
                                let (value, report) = if family == OrthogonalFamily::Legendre {
                                    plan.integrate(&f, Interval::new(-1.0, 1.0).unwrap())
                                } else {
                                    plan.integrate_weighted(&f, canonical())
                                }
                                .unwrap();
                                close(value, expected);
                                assert_eq!(report.evaluations(), n);
                                assert_eq!(report.error_estimate(), None);
                            }
                        }
                    }
                }
            }
            #[test]
            fn extreme_parameters_and_other_families() {
                for (alpha, beta) in [
                    (-0.999, 0.0),
                    (-0.99, -0.99),
                    (100.0, 120.0),
                    (0.0, 1000.0),
                    (0.0, 2000.0),
                ] {
                    for variant in [
                        GaussianRuleVariant::Gauss,
                        GaussianRuleVariant::RadauLeft,
                        GaussianRuleVariant::RadauRight,
                        GaussianRuleVariant::Lobatto,
                    ] {
                        // Mass overflow depends on the selected scalar width.
                        let prepared = GaussianRulePlan::<$s>::with_options(
                            OrthogonalFamily::Jacobi { alpha, beta },
                            8,
                            variant,
                            GaussianRuleOptions::default(),
                        );
                        if beta == 2000.0 || (beta == 1000.0 && core::mem::size_of::<$s>() == 4) {
                            assert!(matches!(
                                (prepared).as_ref().err().unwrap().view(),
                                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                            ));
                            continue;
                        }
                        let plan = prepared.unwrap();
                        let expected_mean = (beta - alpha) / (alpha + beta + 2.0);
                        let mean: $s = plan
                            .nodes()
                            .iter()
                            .zip(plan.weights())
                            .map(|(&x, &w)| x * (w / plan.zeroth_moment()))
                            .sum();
                        close(mean, expected_mean);
                    }
                }
                for family in [
                    OrthogonalFamily::ChebyshevT,
                    OrthogonalFamily::ChebyshevU,
                    OrthogonalFamily::Gegenbauer { lambda: -0.25 },
                    OrthogonalFamily::Gegenbauer { lambda: 1.5 },
                ] {
                    let plan = GaussianRulePlan::<$s>::new(family, 8).unwrap();
                    let mass: $s = match family {
                        OrthogonalFamily::ChebyshevT => core::f64::consts::PI as $s,
                        OrthogonalFamily::ChebyshevU => (core::f64::consts::PI / 2.0) as $s,
                        OrthogonalFamily::Gegenbauer { lambda: 1.5 } => 4.0 / 3.0,
                        _ => plan.zeroth_moment(), // negative parameter qualified by moments below
                    };
                    close(
                        plan.integrate_weighted(|_: $s| Ok(1.0), canonical())
                            .unwrap()
                            .0,
                        mass,
                    );
                    let lambda = match family {
                        OrthogonalFamily::Gegenbauer { lambda } => lambda,
                        OrthogonalFamily::ChebyshevT => 0.0,
                        _ => 1.0,
                    };
                    close(
                        plan.integrate_weighted(|x: $s| Ok(x * x), canonical())
                            .unwrap()
                            .0
                            / mass,
                        1.0 / (2.0 * (lambda + 1.0)),
                    );
                }
            }
            #[test]
            fn endpoint_mapping_and_product_rule() {
                for variant in [
                    GaussianRuleVariant::RadauLeft,
                    GaussianRuleVariant::RadauRight,
                    GaussianRuleVariant::Lobatto,
                ] {
                    let n = if variant == GaussianRuleVariant::Lobatto {
                        2
                    } else {
                        1
                    };
                    let plan = GaussianRulePlan::<$s>::with_options(
                        OrthogonalFamily::Legendre,
                        n,
                        variant,
                        GaussianRuleOptions::default(),
                    )
                    .unwrap();
                    let interval = Interval::new(2.0, 5.0).unwrap();
                    close(plan.integrate(|_: $s| Ok(1.0), interval).unwrap().0, 3.0);
                    let product =
                        ProductRule::new(&plan, scioxide_integrate::ProductOptions::default());
                    let f = (|_: &[$s]| Ok(1.0));
                    close(product.integrate(&f, &[interval, interval]).unwrap().0, 9.0);
                }
            }
        }
    };
}
width_cases!(double, f64, 3.0e-12);
width_cases!(single, f32, 4.0e-5);

#[test]
fn hundred_node_legendre_weights_keep_high_precision_accuracy() {
    let plan = GaussianRulePlan::<f64>::gauss_legendre(100).unwrap();
    let rows = include_str!("../data/gaussian_legendre_100.csv")
        .lines()
        .filter(|row| !row.starts_with('#'));
    assert_eq!(rows.clone().count(), plan.nodes().len());
    for ((&node, &weight), row) in plan.nodes().iter().zip(plan.weights()).zip(rows) {
        let (x, w) = row.split_once(',').unwrap();
        let expected_node: f64 = x.parse().unwrap();
        let expected_weight: f64 = w.parse().unwrap();
        assert!((node - expected_node).abs() < 5e-15);
        assert!(
            (weight / expected_weight - 1.0).abs() < 3e-13,
            "node={node}, relative weight error={}",
            weight / expected_weight - 1.0
        );
    }
}

#[test]
fn large_rules_keep_mass_moments_and_exact_parity() {
    for n in [100, 1000, 2000] {
        for variant in [
            GaussianRuleVariant::Gauss,
            GaussianRuleVariant::RadauLeft,
            GaussianRuleVariant::RadauRight,
            GaussianRuleVariant::Lobatto,
        ] {
            let plan = GaussianRulePlan::<f64>::with_options(
                OrthogonalFamily::Legendre,
                n,
                variant,
                GaussianRuleOptions::default(),
            )
            .unwrap();
            assert!(plan.nodes().windows(2).all(|pair| pair[0] < pair[1]));
            assert!(plan.weights().iter().all(|&w| w.is_finite() && w > 0.0));
            for degree in [0, 1, 2, 9, 20] {
                let moment: f64 = plan
                    .nodes()
                    .iter()
                    .zip(plan.weights())
                    .map(|(&x, &w)| w * x.powi(degree))
                    .sum();
                let expected = if degree % 2 == 0 {
                    2.0 / f64::from(degree + 1)
                } else {
                    0.0
                };
                assert!(
                    (moment - expected).abs() < 3e-13,
                    "n={n}, {variant:?}, degree={degree}: {moment}"
                );
            }
            if matches!(
                variant,
                GaussianRuleVariant::Gauss | GaussianRuleVariant::Lobatto
            ) {
                for i in 0..n / 2 {
                    assert_eq!(plan.nodes()[i], -plan.nodes()[n - 1 - i]);
                    assert_eq!(plan.weights()[i], plan.weights()[n - 1 - i]);
                }
            }
            if matches!(
                variant,
                GaussianRuleVariant::RadauLeft | GaussianRuleVariant::Lobatto
            ) {
                assert_eq!(plan.nodes()[0], -1.0);
            }
            if matches!(
                variant,
                GaussianRuleVariant::RadauRight | GaussianRuleVariant::Lobatto
            ) {
                assert_eq!(plan.nodes()[n - 1], 1.0);
            }
        }
    }
    let plan = GaussianRulePlan::<f32>::gauss_legendre(1000).unwrap();
    assert!((plan.weights().iter().sum::<f32>() - 2.0).abs() < 3e-5);
    assert!(plan.weights().iter().all(|&weight| weight > 0.0));
}

/// Moments `∫ x^k (1-x)^a (1+x)^b dx` for `k <= 4`, from `x = 2u - 1` with
/// `u ~ Beta(b + 1, a + 1)`.
fn jacobi_moments(a: f64, b: f64) -> [f64; 5] {
    let mass =
        2f64.powf(a + b + 1.0) * (lgamma(a + 1.0) + lgamma(b + 1.0) - lgamma(a + b + 2.0)).exp();
    let mut u = [1.0; 5];
    for j in 1..5 {
        u[j] = u[j - 1] * (b + j as f64) / (a + b + 1.0 + j as f64);
    }
    core::array::from_fn(|k| {
        let mut binomial = 1.0;
        let mut moment = 0.0;
        for (j, &uj) in u.iter().enumerate().take(k + 1) {
            let sign = if (k - j) % 2 == 0 { 1.0 } else { -1.0 };
            moment += binomial * 2f64.powi(j as i32) * sign * uj;
            binomial = binomial * (k - j) as f64 / (j + 1) as f64;
        }
        mass * moment
    })
}

/// Lanczos `ln Γ(x)` for `x > 0`, accurate to about 1e-15 relative.
fn lgamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (core::f64::consts::PI / (core::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let series = G[1..]
        .iter()
        .enumerate()
        .fold(G[0], |sum, (i, &g)| sum + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * core::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

#[test]
fn endpoint_tolerance_scales_with_order_and_keeps_moments() {
    // Prescribed endpoints carry the eigensolver's order-dependent backward
    // error; these rules once failed a fixed eight-epsilon endpoint check.
    let cases: &[(f64, GaussianRuleVariant, &[usize])] = &[
        (-0.99, GaussianRuleVariant::RadauRight, &[500, 1000, 2000]),
        (-0.99, GaussianRuleVariant::Lobatto, &[300, 1500]),
        (-0.9, GaussianRuleVariant::RadauRight, &[1500]),
        (0.5, GaussianRuleVariant::RadauLeft, &[2000]),
    ];
    for &(a, variant, orders) in cases {
        let moments = jacobi_moments(a, a);
        for &n in orders {
            let plan = GaussianRulePlan::<f64>::with_options(
                OrthogonalFamily::Jacobi { alpha: a, beta: a },
                n,
                variant,
                GaussianRuleOptions::default(),
            );
            assert!(plan.is_ok(), "a={a}, {variant:?}, n={n}: {plan:?}");
            let plan = plan.unwrap();
            let nodes = plan.nodes();
            assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(plan.weights().iter().all(|&w| w.is_finite() && w > 0.0));
            if variant != GaussianRuleVariant::RadauRight {
                assert_eq!(nodes[0], -1.0);
            }
            if variant != GaussianRuleVariant::RadauLeft {
                assert_eq!(nodes[n - 1], 1.0);
            }
            for (k, &expected) in moments.iter().enumerate() {
                let moment: f64 = nodes
                    .iter()
                    .zip(plan.weights())
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();
                assert!(
                    (moment - expected).abs() <= 1e-12 * moments[0],
                    "a={a}, {variant:?}, n={n}, k={k}: {moment} vs {expected}"
                );
            }
        }
    }
    for (variant, n) in [
        (GaussianRuleVariant::RadauLeft, 2000),
        (GaussianRuleVariant::Lobatto, 1000),
        (GaussianRuleVariant::Lobatto, 2000),
    ] {
        let family = OrthogonalFamily::Jacobi {
            alpha: -0.99_f32,
            beta: -0.99,
        };
        let plan = GaussianRulePlan::<f32>::with_options(
            family,
            n,
            variant,
            GaussianRuleOptions::default(),
        )
        .unwrap();
        assert_eq!(plan.nodes()[0], -1.0);
        assert!(plan.nodes().windows(2).all(|pair| pair[0] < pair[1]));
    }
}

#[test]
fn preparation_options_expose_only_tolerance_and_execution_limits() {
    let tolerance = scioxide_core::Tolerance::from_machine_epsilon();
    let execution = LinalgExecutionOptions::new().with_byte_limit(1);
    let options = GaussianRuleOptions::<f64>::default()
        .with_deflation_tolerance(tolerance)
        .with_execution_options(execution);
    assert_eq!(options.deflation_tolerance(), tolerance);
    assert_eq!(options.execution_options(), execution);
    let error = GaussianRulePlan::with_options(
        OrthogonalFamily::Legendre,
        8,
        GaussianRuleVariant::Gauss,
        options,
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn invalid_combinations_and_preparation_budget_fail_explicitly() {
    for variant in [
        GaussianRuleVariant::Gauss,
        GaussianRuleVariant::RadauLeft,
        GaussianRuleVariant::RadauRight,
        GaussianRuleVariant::Lobatto,
    ] {
        assert!(matches!(
            GaussianRulePlan::<f64>::with_options(
                OrthogonalFamily::Legendre,
                0,
                variant,
                GaussianRuleOptions::default()
            )
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }
    assert!(
        GaussianRulePlan::<f32>::with_options(
            OrthogonalFamily::Legendre,
            1,
            GaussianRuleVariant::Lobatto,
            GaussianRuleOptions::default()
        )
        .is_err()
    );
    assert!(
        GaussianRulePlan::<f64>::with_options(
            OrthogonalFamily::HermiteH,
            4,
            GaussianRuleVariant::RadauLeft,
            GaussianRuleOptions::default()
        )
        .is_err()
    );
    let exhausted = GaussianRulePlan::<f64>::with_options(
        OrthogonalFamily::Legendre,
        32,
        GaussianRuleVariant::Gauss,
        GaussianRuleOptions::default().with_execution_options(
            LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
        ),
    );
    assert!(matches!(
        exhausted.as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    let hermite = GaussianRulePlan::<f64>::new(OrthogonalFamily::HermiteH, 4).unwrap();
    // The unweighted protocol rejects weighted families before any callback.
    assert!(matches!(
        hermite
            .integrate(|_| Ok(1.0), Interval::new(0.0, 0.0).unwrap())
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    for support in [
        GaussianSupport::HalfLine(0.0),
        GaussianSupport::Interval(Interval::new(-1.0, 1.0).unwrap()),
    ] {
        assert!(matches!(
            hermite.integrate_weighted(|_| Ok(1.0), support).err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("support")
        ));
    }
    assert!(matches!(
        hermite
            .integrate_weighted(|_| Ok(f64::NAN), GaussianSupport::RealLine)
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Callback,
            ..
        }
    ));
    let laguerre = GaussianRulePlan::<f32>::new(OrthogonalFamily::Laguerre, 4).unwrap();
    assert!(matches!(
        laguerre
            .integrate_weighted(|_| Ok(1.0), GaussianSupport::HalfLine(f32::NAN))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
    assert!(matches!(
        laguerre
            .integrate_weighted(|_| Ok(1.0), GaussianSupport::RealLine)
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
            if argument == scioxide_core::argument!("support")
    ));
    assert!(GaussianRulePlan::<f32>::new(OrthogonalFamily::HermiteH, 256).is_err());
}
