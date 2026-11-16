#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{Budget, ExtendedInterval, RealScalar, SciOxideError};
use scioxide_random::*;
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a:e} != {b:e}");
}
fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn check<S: RealScalar>() {
    let tol = if core::mem::size_of::<S>() == 4 {
        3e-5
    } else {
        3e-12
    };
    let normal = Normal::new(S::ZERO, S::ONE).unwrap();
    let logistic = Logistic::new(S::ZERO, S::ONE).unwrap();
    let uniform = Uniform::new(S::ZERO, S::ONE).unwrap();
    for p in [1e-30, 1e-10, 0.01, 0.5, 0.9] {
        let p = scalar::<S>(p);
        for law in [&normal as &dyn InverseSurvival<S>, &logistic, &uniform] {
            assert!(law.isf(p).unwrap().is_finite());
        }
        close(
            normal.log_sf(normal.isf(p).unwrap()).unwrap().into_f64(),
            p.ln().into_f64(),
            tol,
        );
        close(
            logistic
                .log_sf(logistic.isf(p).unwrap())
                .unwrap()
                .into_f64(),
            p.ln().into_f64(),
            tol,
        );
        close(
            uniform
                .log_cdf(uniform.quantile(p).unwrap())
                .unwrap()
                .into_f64(),
            p.ln().into_f64(),
            tol,
        );
    }
    let histogram = Histogram::from_masses(
        &[scalar(0.0), scalar(1.0), scalar(2.0), scalar(3.0)],
        &[S::ONE, S::ZERO, S::ONE],
    )
    .unwrap();
    assert!(!histogram.contains(scalar(1.5)).unwrap());
    close(
        histogram.quantile(scalar(0.5)).unwrap().into_f64(),
        1.0,
        tol,
    );
    close(histogram.mean().unwrap().into_f64(), 1.5, tol);
    close(histogram.variance().unwrap().into_f64(), 13.0 / 12.0, tol);
    close(histogram.entropy().unwrap().into_f64(), 2.0_f64.ln(), tol);
    let heights =
        Histogram::from_densities(&[scalar(0.0), scalar(1.0), scalar(3.0)], &[S::ONE, S::ONE])
            .unwrap();
    close(heights.cdf(S::ONE).unwrap().into_f64(), 1.0 / 3.0, tol);
    let affine = Affine::new(&normal, scalar(2.0), scalar(-3.0)).unwrap();
    close(affine.mean().unwrap().into_f64(), 2.0, tol);
    close(affine.variance().unwrap().into_f64(), 9.0, tol);
    close(
        affine.entropy().unwrap().into_f64(),
        normal.entropy().unwrap().into_f64() + 3.0_f64.ln(),
        tol,
    );
    for x in [-2.0, 0.0, 2.0] {
        let x = scalar::<S>(x);
        close(
            affine.cdf(x).unwrap().into_f64(),
            normal
                .sf((x - scalar(2.0)) / scalar(-3.0))
                .unwrap()
                .into_f64(),
            tol,
        );
    }
    let truncated = Truncated::new(&normal, scalar(10.0), scalar(11.0)).unwrap();
    for p in [1e-4, 0.1, 0.5, 0.9] {
        let p = scalar::<S>(p);
        let x = truncated.quantile(p).unwrap();
        close(
            truncated.cdf(x).unwrap().into_f64(),
            p.into_f64(),
            tol * 20.0,
        );
        let y = truncated.isf(p).unwrap();
        close(
            truncated.sf(y).unwrap().into_f64(),
            p.into_f64(),
            tol * 20.0,
        );
    }
    let folded = Transformed::abs(&normal).unwrap();
    close(
        folded.pdf(S::ONE).unwrap().into_f64(),
        2.0 * normal.pdf(S::ONE).unwrap().into_f64(),
        tol,
    );
    close(
        folded.sf(S::ONE).unwrap().into_f64(),
        2.0 * normal.sf(S::ONE).unwrap().into_f64(),
        tol,
    );
    let exponential = Transformed::exp(&normal).unwrap();
    let original = LogNormal::new(S::ZERO, S::ONE).unwrap();
    for x in [0.01, 0.5, 1.0, 10.0] {
        let x = scalar::<S>(x);
        close(
            exponential.log_pdf(x).unwrap().into_f64(),
            original.log_pdf(x).unwrap().into_f64(),
            tol,
        );
    }
    let recovered = Transformed::log(&exponential).unwrap();
    close(
        recovered.log_cdf(scalar(-2.0)).unwrap().into_f64(),
        normal.log_cdf(scalar(-2.0)).unwrap().into_f64(),
        tol,
    );
    let order = OrderStatistic::new(&uniform, 3, 8).unwrap();
    let beta = Beta::new(scalar(3.0), scalar(6.0)).unwrap();
    for x in [0.01, 0.3, 0.9] {
        let x = scalar::<S>(x);
        close(
            order.log_pdf(x).unwrap().into_f64(),
            beta.log_pdf(x).unwrap().into_f64(),
            tol,
        );
        close(
            order.log_sf(x).unwrap().into_f64(),
            beta.log_sf(x).unwrap().into_f64(),
            tol,
        );
    }
    let components: Vec<&dyn ContinuousLaw<S>> = vec![&normal, &logistic];
    let mixture = Mixture::new(components, &[S::ONE, S::ONE]).unwrap();
    close(
        mixture.pdf(S::ZERO).unwrap().into_f64(),
        (normal.pdf(S::ZERO).unwrap().into_f64() + 0.25) / 2.0,
        tol,
    );
    for p in [1e-10, 0.1, 0.5, 0.9] {
        let p = scalar::<S>(p);
        let q = mixture.isf(p).unwrap();
        close(
            mixture.log_sf(q).unwrap().into_f64(),
            p.ln().into_f64(),
            tol * 10.0,
        );
    }
    let count = Binomial::new(20, scalar::<S>(0.3)).unwrap();
    close(
        (0..=20).map(|k| count.pmf(k).unwrap().into_f64()).sum(),
        1.0,
        tol,
    );
    close(count.mean().unwrap().into_f64(), 6.0, tol);
    for p in [1e-10, 0.1, 0.5, 0.99] {
        let p = scalar::<S>(p);
        let k = count.quantile(p).unwrap();
        assert!(count.cdf(k).unwrap() >= p);
        if k > 0 {
            assert!(count.cdf(k - 1).unwrap() < p);
        }
        let k = count.isf(p).unwrap();
        assert!(count.sf(k).unwrap() <= p);
        if k > 0 {
            assert!(count.sf(k - 1).unwrap() > p);
        }
    }
}
#[test]
fn f64_contract() {
    check::<f64>();
}
#[test]
fn f32_contract() {
    check::<f32>();
}
#[test]
fn generalized_inverse_gaps() {
    let a = Uniform::new(0.0, 1.0).unwrap();
    let b = Uniform::new(2.0, 3.0).unwrap();
    let mixture = Mixture::new(vec![a, b], &[1.0, 1.0]).unwrap();
    close(mixture.quantile(0.5).unwrap(), 1.0, 1e-12);
    assert!(!mixture.contains(1.5).unwrap());
}
#[test]
fn explicit_failures_and_budgets() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    assert!(Logistic::new(0.0, -1.0).is_err());
    assert!(Logistic::new(f64::NAN, 1.0).is_err());
    assert!(Affine::new(&normal, 0.0, 0.0).is_err());
    let crossing = Histogram::from_masses(&[-1.0, 1.0], &[1.0]).unwrap();
    assert!(Transformed::log(crossing).is_err());
    assert!(Truncated::new(&normal, 1.0, 1.0).is_err());
    assert!(OrderStatistic::new(&normal, 0, 2).is_err());
    assert!(Histogram::from_masses(&[0.0, 1.0], &[0.0]).is_err());
    assert!(Histogram::from_masses(&[0.0, 0.0], &[1.0]).is_err());
    assert!(Mixture::new(vec![normal.clone()], &[-1.0]).is_err());
    let options = InversionOptions::default()
        .with_root_options(scioxide_rootfind::ScalarRootOptions::default())
        .with_work_budget(Budget::ONE);
    let error = invert_log_probability_with_options(&normal, -10.0, true, &options).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let error = Binomial::with_options(
        20,
        0.3,
        scioxide_random::LawOptions::default().with_work_budget(Budget::ONE),
    )
    .unwrap()
    .entropy()
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    close(
        Binomial::new(1, 0.5).unwrap().entropy().unwrap(),
        2.0_f64.ln(),
        1e-14,
    );
    let report =
        invert_log_probability_with_options(&normal, -1000.0, true, &InversionOptions::default())
            .unwrap();
    assert!(report.root().is_success());
    assert!(report.evaluations() <= 4096);
    close(normal.log_sf(report.value()).unwrap(), -1000.0, 1e-6);
    let tiny = Uniform::new(0.0, f64::from_bits(2)).unwrap();
    assert!(tiny.entropy().unwrap().is_finite());
    assert!(tiny.log_cdf(f64::from_bits(1)).unwrap().is_finite());
}
struct Provider;
impl Support<f64> for Provider {
    fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
        ExtendedInterval::new(Some(0.0), Some(1.0))
    }
}
impl Density<f64> for Provider {
    fn pdf(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 })
    }
}
#[test]
fn typed_provider() {
    let law = Provider;
    assert_eq!(law.pdf(0.5).unwrap(), 1.0);
}
#[test]
fn inverse_sampling_transaction() {
    let law = Truncated::new(Normal::new(0.0, 1.0).unwrap(), 8.0, 9.0).unwrap();
    let mut a = SplitMix64::from_seed(531);
    let mut b = a.clone();
    let policy = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
    let owned = law.sample_n(&mut a, 8).unwrap();
    let mut output = [0.0; 8];
    let mut workspace = policy
        .workspace(scioxide_random::SamplingRequest::scalars(8))
        .unwrap();
    law.sample_n_into(&mut b, policy, &mut output, &mut workspace)
        .unwrap();
    assert_eq!(owned, output);
    assert!(output.iter().all(|x| *x >= 8.0 && *x <= 9.0));
}

fn fixtures<S: RealScalar>() {
    assert_eq!(include_str!("../data/law_contract.tsv").lines().count(), 22);
    let tol = if core::mem::size_of::<S>() == 4 {
        8e-5
    } else {
        5e-12
    };
    let normal = Normal::new(S::ZERO, S::ONE).unwrap();
    let logistic = Logistic::new(S::ZERO, S::ONE).unwrap();
    let binomial = Binomial::new(20, scalar::<S>(0.3)).unwrap();
    for line in include_str!("../data/law_contract.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let x: f64 = f[1].parse().unwrap();
        let expected: Vec<f64> = f[2..].iter().map(|v| v.parse().unwrap()).collect();
        let actual = match f[0] {
            "normal" => [
                normal.log_pdf(scalar(x)).unwrap(),
                normal.log_cdf(scalar(x)).unwrap(),
                normal.log_sf(scalar(x)).unwrap(),
            ],
            "logistic" => [
                logistic.log_pdf(scalar(x)).unwrap(),
                logistic.log_cdf(scalar(x)).unwrap(),
                logistic.log_sf(scalar(x)).unwrap(),
            ],
            _ => {
                let k: usize = f[1].parse().unwrap();
                [
                    binomial.log_pmf(k).unwrap(),
                    binomial.log_cdf(k).unwrap(),
                    binomial.log_sf(k).unwrap(),
                ]
            }
        };
        for (a, b) in actual.into_iter().zip(expected) {
            close(a.into_f64(), b, tol);
        }
    }
}
#[test]
fn independent_f64_fixtures() {
    fixtures::<f64>();
}
#[test]
fn independent_f32_fixtures() {
    fixtures::<f32>();
}
#[test]
fn density_normalization() {
    let logistic = Logistic::new(0.0, 1.0).unwrap();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let trunc = Truncated::new(&normal, 10.0, 11.0).unwrap();
    let order = OrderStatistic::new(Uniform::new(0.0, 1.0).unwrap(), 3, 8).unwrap();
    // Independent composite midpoint rule with a fixed fine mesh, test-only.
    for (law, low, high) in [
        (&logistic as &dyn Density<f64>, -30.0, 30.0),
        (&trunc, 10.0, 11.0),
        (&order, 0.0, 1.0),
    ] {
        let n = 10000;
        let width = (high - low) / f64::from(n);
        let total = (0..n)
            .map(|i| law.pdf(low + (f64::from(i) + 0.5) * width).unwrap() * width)
            .sum();
        close(total, 1.0, 1e-6);
    }
}

#[test]
fn additional_tail_and_support_boundaries() {
    let c = Cauchy::new(-f64::MAX, 1.0).unwrap();
    assert!(c.log_pdf(f64::MAX).unwrap().is_finite());
    close(
        c.log_sf(f64::MAX).unwrap(),
        -f64::MAX.ln() - 2.0_f64.ln() - core::f64::consts::PI.ln(),
        1e-14,
    );
    let t = StudentT::new(1.0).unwrap();
    let c = Cauchy::new(0.0, 1.0).unwrap();
    close(
        t.log_sf(f64::MAX).unwrap(),
        c.log_sf(f64::MAX).unwrap(),
        1e-12,
    );
    for p in [0.0, 0.5, 1.0] {
        assert_eq!(Binomial::new(20, 1.0).unwrap().quantile(p).unwrap(), 20);
        assert_eq!(Binomial::new(20, 0.0).unwrap().isf(p).unwrap(), 0);
    }
    let mix = Mixture::new(
        vec![
            Uniform::new(0.0, 1.0).unwrap(),
            Uniform::new(2.0, 3.0).unwrap(),
        ],
        &[1.0, 1.0],
    )
    .unwrap();
    close(mix.isf(0.5).unwrap(), 2.0, 1e-12);
    let reflected = Affine::new(mix, 0.0, -1.0).unwrap();
    close(reflected.quantile(0.5).unwrap(), -2.0, 1e-12);
    let gamma = Gamma::new(2.0, 1.0).unwrap();
    let beta = Beta::new(2.0, 3.0).unwrap();
    for p in [1e-20, 0.1, 0.5] {
        close(gamma.log_sf(gamma.isf(p).unwrap()).unwrap(), p.ln(), 1e-10);
        close(beta.log_sf(beta.isf(p).unwrap()).unwrap(), p.ln(), 1e-8);
    }
}

#[test]
fn histogram_partial_bin_log_probability_survives_ratio_underflow() {
    let law = Histogram::from_masses(&[0.0_f64, 1e300], &[1.0]).unwrap();
    assert_eq!(law.cdf(1e-300).unwrap(), 0.0);
    close(law.log_cdf(1e-300).unwrap(), -600.0 * 10.0_f64.ln(), 1e-14);
    let law = Histogram::from_masses(&[-1e300_f64, 0.0], &[1.0]).unwrap();
    close(law.log_sf(-1e-300).unwrap(), -600.0 * 10.0_f64.ln(), 1e-14);
    let law = Histogram::from_masses(&[0.0_f32, 1e30], &[1.0]).unwrap();
    close(
        f64::from(law.log_cdf(1e-30).unwrap()),
        -60.0 * 10.0_f64.ln(),
        1e-6,
    );
}

#[test]
fn transforms_trim_support_gaps_at_endpoints() {
    let histogram = Histogram::from_masses(&[-3.0, -2.0, 2.0, 3.0], &[1.0, 0.0, 1.0]).unwrap();
    let folded = Transformed::abs(&histogram).unwrap();
    assert_eq!(folded.quantile(0.0).unwrap(), 2.0);
    assert_eq!(
        folded.support().unwrap(),
        ExtendedInterval::new(Some(2.0), Some(3.0)).unwrap()
    );
    let clipped = Truncated::new(&histogram, -1.0, 2.5).unwrap();
    assert_eq!(clipped.quantile(0.0).unwrap(), 2.0);
    let clipped = Truncated::new(&histogram, -2.0, 2.5).unwrap();
    assert_eq!(clipped.quantile(0.0).unwrap(), 2.0);
    let reflected = Affine::new(&histogram, 0.0, -2.0).unwrap();
    let clipped = Truncated::new(&reflected, -2.0, 5.0).unwrap();
    assert_eq!(clipped.quantile(0.0).unwrap(), 4.0);
    let positive = Histogram::from_masses(&[1.0, 2.0, 4.0, 5.0], &[1.0, 0.0, 1.0]).unwrap();
    let transformed = Transformed::log(&positive).unwrap();
    let clipped = Truncated::new(transformed, 3.0_f64.ln(), 4.5_f64.ln()).unwrap();
    close(clipped.quantile(0.0).unwrap(), 4.0_f64.ln(), 1e-14);
}

#[test]
fn law_function_forms_agree_bit_for_bit() {
    use scioxide_core::ExecutionContext;
    let normal = Normal::<f64>::standard();
    let options = LawOptions::default();
    for (low, high) in [
        (-1.0, 1.0),
        (2.0, 40.0),
        (f64::NEG_INFINITY, 0.5),
        (3.0, f64::INFINITY),
        (0.5, 0.5),
    ] {
        let owned = log_interval_probability(&normal, low, high).unwrap();
        let configured =
            log_interval_probability_with_options(&normal, low, high, options).unwrap();
        let scoped = log_interval_probability_scoped(
            &normal,
            low,
            high,
            options,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(owned.to_bits(), configured.to_bits());
        assert_eq!(owned.to_bits(), scoped.to_bits());
    }
    let inversion = InversionOptions::default();
    for (log_p, upper) in [(0.25_f64.ln(), false), (-1000.0, true), (-1e-3, false)] {
        let report =
            invert_log_probability_with_options(&normal, log_p, upper, &inversion).unwrap();
        let scoped = invert_log_probability_scoped(
            &normal,
            log_p,
            upper,
            &inversion,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        let owned = invert_log_probability(&normal, log_p, upper).unwrap();
        assert!(report.root().is_success());
        assert_eq!(owned.to_bits(), report.value().to_bits());
        assert_eq!(scoped.value().to_bits(), report.value().to_bits());
        assert_eq!(scoped.evaluations(), report.evaluations());
        assert_eq!(scoped.root().termination(), report.root().termination());
    }
}

/// Runs `body` inside a parent solve whose evaluation limit is `limit`.
fn in_parent<T>(
    limit: usize,
    body: impl FnOnce(&mut scioxide_core::SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    use scioxide_core::{
        BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveScope,
        SolveState,
    };
    SolveScope::from(ExecutionContext::sequential()).run(
        &mut SolveState::new(),
        SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "parent", 1),
            0,
            0,
        ),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(limit)?),
        body,
    )
}

#[test]
fn count_inverses_share_parent_evaluation_budgets() {
    fn check<T: core::fmt::Debug + PartialEq, D: Quantile<f64, T> + InverseSurvival<f64, T>>(
        law: D,
    ) {
        for upper in [false, true] {
            let query = |context: &mut scioxide_core::SolveContext<'_, '_>| {
                if upper {
                    law.isf_scoped(0.4, context)
                } else {
                    law.quantile_scoped(0.4, context)
                }
            };
            assert_eq!(
                in_parent(1, query).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::ConvergenceFailure
            );
            let expected = if upper {
                law.isf(0.4)
            } else {
                law.quantile(0.4)
            }
            .unwrap();
            let actual = in_parent(1_000_000, |context| {
                let value = query(context)?;
                assert!(
                    context
                        .budget_report(scioxide_core::BudgetKind::Evaluation)
                        .consumed()
                        > 1
                );
                Ok(value)
            })
            .unwrap();
            assert_eq!(actual, expected);
        }
    }
    check(Binomial::new(100, 0.3).unwrap());
    check(Poisson::new(30.0).unwrap());
    check::<usize, &Poisson<f64>>(&Poisson::new(30.0).unwrap());
    check(PoissonBinomial::new(&[0.3; 10]).unwrap());
    check(DiscreteLaplace::new(0.1).unwrap());
    check(Skellam::new(3.0, 2.0).unwrap());
}

#[test]
fn iterative_count_moments_and_entropy_share_parent_budgets() {
    fn check<D: Mean<f64> + Variance<f64>>(law: D) {
        assert!(in_parent(1, |context| law.mean_scoped(context)).is_err());
        assert!(in_parent(1, |context| law.variance_scoped(context)).is_err());
        in_parent(1_000_000, |context| {
            close(law.mean_scoped(context)?, law.mean()?, 1e-13);
            close(law.variance_scoped(context)?, law.variance()?, 1e-13);
            assert!(
                context
                    .budget_report(scioxide_core::BudgetKind::Evaluation)
                    .consumed()
                    > 2
            );
            Ok(())
        })
        .unwrap();
    }
    check(Zipf::new(4.0_f64).unwrap());
    check(Zipfian::new(2.0_f64, 100).unwrap());
    let law = Binomial::new(100, 0.3_f64).unwrap();
    assert!(in_parent(1, |context| law.entropy_scoped(context)).is_err());
    assert!(in_parent(1, |context| Entropy::entropy_scoped(&&law, context)).is_err());
    close(
        in_parent(101, |context| law.entropy_scoped(context)).unwrap(),
        law.entropy().unwrap(),
        1e-13,
    );
}

#[test]
fn poisson_point_mass_inverse_endpoints_are_finite() {
    let law = Poisson::new(0.0_f64).unwrap();
    for p in [0.0, 0.4, 1.0] {
        assert_eq!(law.quantile(p).unwrap(), 0);
        assert_eq!(law.isf(p).unwrap(), 0);
    }
}

fn scoped_formula_validation<S: RealScalar>() {
    use scioxide_core::SciOxideErrorKind;
    let law = Pareto::new(S::TWO).unwrap();
    for p in [-S::ONE, S::TWO, S::NAN, S::INFINITY, S::NEG_INFINITY] {
        let expected = if p.is_finite() {
            SciOxideErrorKind::DomainError
        } else {
            SciOxideErrorKind::NonFinite
        };
        for result in [
            law.quantile(p),
            in_parent(10, |context| law.quantile_scoped(p, context)),
            law.isf(p),
            in_parent(10, |context| law.isf_scoped(p, context)),
        ] {
            assert_eq!(result.unwrap_err().kind(), expected);
        }
    }
    // Interior inverses that round to the support endpoint retain their error.
    let concentrated = Pareto::new(S::MAX).unwrap();
    let plain = concentrated.quantile(S::HALF).unwrap_err();
    let scoped =
        in_parent(10, |context| concentrated.quantile_scoped(S::HALF, context)).unwrap_err();
    assert_eq!(plain.kind(), scoped.kind());
    // Valid queries agree and each closed-form inverse charges exactly once.
    in_parent(2, |context| {
        assert_eq!(
            law.quantile(S::HALF)?,
            law.quantile_scoped(S::HALF, context)?
        );
        assert_eq!(law.isf(S::HALF)?, law.isf_scoped(S::HALF, context)?);
        assert_eq!(
            context
                .budget_report(scioxide_core::BudgetKind::Evaluation)
                .consumed(),
            2
        );
        assert_eq!(
            law.quantile_scoped(S::HALF, context).unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn formula_scoped_inverses_preserve_validation_and_work() {
    scoped_formula_validation::<f32>();
    scoped_formula_validation::<f64>();
}

#[test]
fn formula_scoped_samples_preserve_range_errors_and_rng_position() {
    let law = Pareto::new(f64::MIN_POSITIVE).unwrap();
    let mut plain_rng = SplitMix64::from_seed(7);
    let mut scoped_rng = plain_rng.clone();
    let plain = law.sample(&mut plain_rng).unwrap_err();
    let scoped = in_parent(10, |context| law.sample_scoped(&mut scoped_rng, context)).unwrap_err();
    assert_eq!(plain.kind(), scoped.kind());
    assert_eq!(plain_rng.position(), scoped_rng.position());
}

#[test]
fn formula_scoped_density_does_not_publish_nan() {
    let law = WeibullMin::new(f64::MAX).unwrap();
    let plain = law.log_pdf(4.0);
    let scoped = in_parent(10, |context| law.log_pdf_scoped(4.0, context));
    assert_eq!(
        plain.map_err(|error| error.kind()),
        scoped.map_err(|error| error.kind())
    );
}

#[test]
fn scoped_law_functions_count_in_the_parent_and_observe_its_limits() {
    use scioxide_core::{BudgetKind, CancellationToken, ExecutionContext, SciOxideErrorKind};
    use std::cell::Cell;
    // Standard exponential law whose tails count their calls.
    let calls = Cell::new(0);
    let law = LogTailsFn::new(
        |x: f64| {
            calls.set(calls.get() + 1);
            Ok((-(-x).exp_m1()).ln())
        },
        |x: f64| {
            calls.set(calls.get() + 1);
            Ok(-x)
        },
        ExtendedInterval::new(Some(0.0), None).unwrap(),
    );
    let inversion = InversionOptions::default();
    let options = LawOptions::default();
    let (median, interval) = in_parent(10_000, |parent| {
        let report =
            invert_log_probability_scoped(&law, 0.5_f64.ln(), false, &inversion, &mut *parent)?;
        let consumed = parent.budget_report(BudgetKind::Evaluation).consumed();
        assert_eq!(consumed, report.evaluations());
        assert_eq!(consumed, calls.get());
        let interval = log_interval_probability_scoped(&law, 1.0, 2.0, options, &mut *parent)?;
        let grown = parent.budget_report(BudgetKind::Evaluation).consumed();
        assert_eq!(grown - consumed, 3); // ln F(2), then ln P(X > 1) and ln P(X > 2)
        assert_eq!(grown, calls.get());
        Ok((report.value(), interval))
    })
    .unwrap();
    assert!((median - core::f64::consts::LN_2).abs() < 1e-12);
    let expected = ((-1.0_f64).exp() - (-2.0_f64).exp()).ln();
    assert!((interval - expected).abs() < 1e-14);
    // A parent limit of one evaluation ends both forms with exhaustion.
    for result in [
        in_parent(1, |parent| {
            invert_log_probability_scoped(&law, 0.5_f64.ln(), false, &inversion, parent)
                .map(|report| report.value())
        }),
        in_parent(1, |parent| {
            log_interval_probability_scoped(&law, 1.0, 2.0, options, parent)
        }),
    ] {
        assert_eq!(
            result.unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
    }
    // A cancelled context stops both forms before any tail query.
    calls.set(0);
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    let error = invert_log_probability_scoped(&law, 0.5_f64.ln(), false, &inversion, cancelled)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    let error = log_interval_probability_scoped(&law, 1.0, 2.0, options, cancelled).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(calls.get(), 0);
}

#[test]
fn nested_kernels_charge_the_shared_budget() {
    fn consumed<T>(
        query: impl FnOnce(&mut scioxide_core::SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> (T, usize) {
        in_parent(1_000_000, |context| {
            let value = query(context)?;
            let used = context
                .budget_report(scioxide_core::BudgetKind::Evaluation)
                .consumed();
            Ok((value, used))
        })
        .unwrap()
    }
    // Argus inverts p > 1/2 through the opposite tail; that work is charged.
    let argus = Argus::new(1.0_f64).unwrap();
    for upper in [false, true] {
        let (value, used) = consumed(|context| {
            if upper {
                argus.isf_scoped(0.9, context)
            } else {
                argus.quantile_scoped(0.9, context)
            }
        });
        let expected = if upper {
            argus.isf(0.9)
        } else {
            argus.quantile(0.9)
        };
        assert_eq!(value, expected.unwrap());
        assert!(used > 0, "upper = {upper}");
        let short = in_parent(used - 1, |context| {
            if upper {
                argus.isf_scoped(0.9, context)
            } else {
                argus.quantile_scoped(0.9, context)
            }
        });
        assert!(short.is_err(), "upper = {upper}");
    }
    // The GIG variance charges its mean as well as its second moment.
    let gig = GeneralizedInverseGaussian::new(0.7_f64, 1.3).unwrap();
    let (mean, mean_work) = consumed(|context| gig.mean_scoped(context));
    let (variance, variance_work) = consumed(|context| gig.variance_scoped(context));
    close(mean, gig.mean().unwrap(), 1e-15);
    close(variance, gig.variance().unwrap(), 1e-15);
    assert!(mean_work > 0);
    assert!(variance_work > mean_work, "{variance_work} <= {mean_work}");
    assert!(in_parent(variance_work - 1, |context| gig.variance_scoped(context)).is_err());
}
