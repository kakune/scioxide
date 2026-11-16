//! Independent rational identities, prepared execution and failure boundaries.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_poly::*;

fn exercise<C: NumericScalar>() {
    let c = |v: f64| C::from_real(C::Real::try_from_f64_lossy(v).unwrap());
    let policy = PolynomialApproximationPolicy::default();
    let p = |v: &[f64]| {
        ApproxPolynomial::monomial_with_options(v.iter().map(|&x| c(x)).collect(), policy).unwrap()
    };
    let linear = p(&[-1.0, 1.0]);
    let quadratic = p(&[1.0, 0.0, 1.0]);
    let denominator = linear
        .try_mul(&linear)
        .unwrap()
        .try_mul(&quadratic)
        .unwrap();
    let plan =
        ApproxPartialFractionPlan::new(&denominator, &[(&linear, 2), (&quadratic, 1)]).unwrap();
    // 1/(x-1) + 2/(x-1)^2 + (3+4x)/(x²+1):
    // numerator = (x-1)(x²+1) + 2(x²+1) + (3+4x)(x-1)²
    //           = 4 - x - 4x² + 5x³.
    let numerator = [c(4.0), c(-1.0), c(-4.0), c(5.0)];
    let result = plan.decompose(&numerator).unwrap();
    assert_eq!(result.polynomial(), &[c(0.0)]);
    let tolerance = C::Real::try_from_usize_exact(128).unwrap() * C::Real::EPSILON;
    for (&actual, expected) in result.residues().iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!(
            (actual - c(expected)).abs() < tolerance,
            "{actual:?} vs {expected}"
        );
    }
    assert_eq!(plan.terms()[0].power(), 1);
    assert_eq!(plan.terms()[1].power(), 2);
    assert_eq!(plan.terms()[2].coefficients(), 2..4);
    assert_eq!(plan.factors()[0].multiplicity(), 2);
    assert!(plan.factors()[0].pole_condition().is_some());
    assert!(plan.factors()[1].pole_condition().is_none());
    assert!(result.report().unwrap().reconstruction_error() < tolerance);
    // Improper numerator adds x+2 times the denominator.
    let improper = p(&[2.0, 1.0])
        .try_mul(&denominator)
        .unwrap()
        .try_add(&p(&[4.0, -1.0, -4.0, 5.0]))
        .unwrap();
    let result = plan.decompose(improper.basis().coefficients()).unwrap();
    assert_eq!(result.polynomial(), &[c(2.0), c(1.0)]);
    for (&actual, expected) in result.residues().iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - c(expected)).abs() < tolerance);
    }
    let zero = plan.decompose(&[c(0.0)]).unwrap();
    assert!(zero.residues().iter().all(|&x| x == c(0.0)));
    let unit = p(&[1.0]);
    let constant = ApproxPartialFractionPlan::new(&unit, &[])
        .unwrap()
        .decompose(&[c(2.0), c(3.0)])
        .unwrap();
    assert_eq!(constant.polynomial(), &[c(2.0), c(3.0)]);
    assert!(constant.residues().is_empty());
    assert!(constant.report().unwrap().reciprocal_condition().is_none());
}
#[test]
fn f32_factors() {
    exercise::<f32>();
}
#[test]
fn f64_factors() {
    exercise::<f64>();
}
#[test]
fn complex32_factors() {
    exercise::<Complex<f32>>();
}
#[test]
fn complex64_factors() {
    exercise::<Complex<f64>>();
}

fn repeated_quadratic<C: NumericScalar>() {
    let c = |x| C::from_real(C::Real::try_from_f64_lossy(x).unwrap());
    let p = |values: &[f64]| {
        ApproxPolynomial::monomial(values.iter().map(|&x| c(x)).collect()).unwrap()
    };
    let denominator = p(&[1., 0., 2., 0., 1.]);
    let quadratic = p(&[1., 0., 1.]);
    let plan = ApproxPartialFractionPlan::new(&denominator, &[(&quadratic, 2)]).unwrap();
    // Independent expansion: (3+4x)/(1+x²) + (5+6x)/(1+x²)².
    let result = plan.decompose(&[c(8.), c(10.), c(3.), c(4.)]).unwrap();
    let tolerance = C::Real::try_from_usize_exact(128).unwrap() * C::Real::EPSILON;
    assert_eq!(result.polynomial(), &[c(0.)]);
    for (&actual, expected) in result.residues().iter().zip([3., 4., 5., 6.]) {
        assert!((actual - c(expected)).abs() <= tolerance * c(expected).abs());
    }
    assert_eq!(plan.terms()[0].power(), 1);
    assert_eq!(plan.terms()[1].power(), 2);
    assert!(result.report().unwrap().reconstruction_error() <= tolerance);
}

#[test]
fn repeated_quadratic_residues_in_all_numeric_domains() {
    repeated_quadratic::<f32>();
    repeated_quadratic::<f64>();
    repeated_quadratic::<Complex<f32>>();
    repeated_quadratic::<Complex<f64>>();
}

fn conjugates<C: NumericScalar>() {
    let policy = PolynomialApproximationPolicy::default();
    let c = |re, im| {
        C::from_components(
            C::Real::try_from_f64_lossy(re).unwrap(),
            C::Real::try_from_f64_lossy(im).unwrap(),
        )
    };
    let p = |v: Vec<C>| ApproxPolynomial::monomial_with_options(v, policy).unwrap();
    let positive = p(vec![c(0.0, -1.0), c(1.0, 0.0)]);
    let negative = p(vec![c(0.0, 1.0), c(1.0, 0.0)]);
    let denominator = positive.try_mul(&negative).unwrap();
    let plan =
        ApproxPartialFractionPlan::new(&denominator, &[(&positive, 1), (&negative, 1)]).unwrap();
    let result = plan.decompose(&[c(1.0, 0.0)]).unwrap();
    let tolerance = C::Real::try_from_usize_exact(16).unwrap() * C::Real::EPSILON;
    assert!((result.residues()[0] - c(0.0, -0.5)).abs() < tolerance);
    assert!((result.residues()[1] - c(0.0, 0.5)).abs() < tolerance);
    let z = c(2.0, 1.0);
    let reconstructed =
        result.residues()[0] / (z - c(0.0, 1.0)) + result.residues()[1] / (z - c(0.0, -1.0));
    assert!((reconstructed - c(1.0, 0.0) / (z * z + c(1.0, 0.0))).abs() < tolerance);
}
#[test]
fn complex_conjugate_residues_pair() {
    conjugates::<Complex<f32>>();
    conjugates::<Complex<f64>>();
}

#[test]
fn checked_factors_reject_invalid_and_unresolved_data() {
    let policy = PolynomialApproximationPolicy::default();
    let p = |v| ApproxPolynomial::monomial_with_options(v, policy).unwrap();
    let factor = p(vec![-1.0_f64, 1.0]);
    let square = factor.try_mul(&factor).unwrap();
    assert!(ApproxPartialFractionPlan::new(&square, &[(&factor, 1), (&factor, 1)]).is_err());
    assert!(ApproxPartialFractionPlan::new(&square, &[(&factor, 0)]).is_err());
    assert!(ApproxPartialFractionPlan::new(&square, &[(&factor, usize::MAX)]).is_err());
    let wrong = p(vec![-2.0, 1.0]);
    assert!(ApproxPartialFractionPlan::new(&square, &[(&wrong, 2)]).is_err());
    let nonmonic = p(vec![-2.0, 2.0]);
    assert!(ApproxPartialFractionPlan::new(&square, &[(&nonmonic, 2)]).is_err());
    let constant = p(vec![1.0]);
    assert!(ApproxPartialFractionPlan::new(&square, &[(&constant, 2)]).is_err());
    let plan = ApproxPartialFractionPlan::new(&square, &[(&factor, 2)]).unwrap();
    assert!(matches!(
        (plan.decompose(&[f64::NAN])).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(plan.decompose(&[]).is_err());
}

#[test]
fn clustered_distinct_poles_preserve_multiplicity() {
    let policy = PolynomialApproximationPolicy::default();
    let p = |root: f64| ApproxPolynomial::monomial_with_options(vec![-root, 1.0], policy).unwrap();
    let a = p(1.0_f64);
    let b = p(1.0 + 1e-6);
    let d = a.try_mul(&b).unwrap();
    let plan = ApproxPartialFractionPlan::new(&d, &[(&a, 1), (&b, 1)]).unwrap();
    let result = plan.decompose(&[1.0]).unwrap();
    let expected = 1.0 / (b.basis().coefficients()[0] - a.basis().coefficients()[0]);
    assert!((result.residues()[0] / expected - 1.0).abs() < 1e-8);
    assert!((result.residues()[1] / -expected - 1.0).abs() < 1e-8);
    assert_eq!(plan.factors()[0].multiplicity(), 1);
    assert_eq!(plan.factors()[1].multiplicity(), 1);
    assert!(
        result
            .report()
            .unwrap()
            .reciprocal_condition()
            .is_some_and(|r| r < 1e-5)
    );
    assert!(result.report().unwrap().reconstruction_error() < 1e-14);
}

#[test]
fn transactional_reusable_output() {
    use scioxide_core::ExecutionContext;
    let policy = PolynomialApproximationPolicy::default();
    let factor = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let plan = ApproxPartialFractionPlan::new(&factor, &[(&factor, 1)]).unwrap();
    let expected = plan.decompose(&[2.0]).unwrap();
    let mut output = expected.clone();
    let mut workspace = PartialFractionWorkspace::new();
    let cx = ExecutionContext::sequential();
    let requirements = plan.workspace_requirements(8).unwrap();
    assert_eq!(requirements.degree(), 1);
    assert!(requirements.required_bytes().unwrap() > 0);
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    let error = plan
        .decompose_into(&[3.0], &mut output, &mut workspace, cx)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, expected);
    workspace.try_reserve(&requirements).unwrap();
    // Storage reserved for longer numerators serves shorter ones.
    assert_eq!(
        workspace
            .check(&plan.workspace_requirements(2).unwrap())
            .unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    let too_long = plan
        .decompose_into(&[1.0; 9], &mut output, &mut workspace, cx)
        .unwrap_err();
    assert!(matches!(
        too_long.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, expected);
    for _ in 0..2 {
        plan.decompose_into(&[2.0], &mut output, &mut workspace, cx)
            .unwrap();
        assert_eq!(output, expected);
    }
    assert!(
        plan.decompose_into(&[f64::INFINITY], &mut output, &mut workspace, cx)
            .is_err()
    );
    assert_eq!(output, expected);
    let mut empty = ApproxPartialFractions::new();
    let error = plan
        .decompose_into(&[2.0], &mut empty, &mut workspace, cx)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(empty.report().is_none());
    // A workspace prepared for another degree needs growth before reuse.
    let quadratic = factor.try_mul(&factor).unwrap();
    let other = ApproxPartialFractionPlan::new(&quadratic, &[(&factor, 2)]).unwrap();
    let other_requirements = other.workspace_requirements(1).unwrap();
    assert_eq!(
        workspace.check(&other_requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    workspace.try_reserve(&other_requirements).unwrap();
    let mut other_output = other.decompose(&[1.0]).unwrap();
    other
        .decompose_into(&[2.0], &mut other_output, &mut workspace, cx)
        .unwrap();
    assert_eq!(other_output, other.decompose(&[2.0]).unwrap());
}

fn fixture_cases<C: NumericScalar>(single: bool) {
    let c = |v: f64| C::from_real(C::Real::try_from_f64_lossy(v).unwrap());
    let policy = PolynomialApproximationPolicy::default();
    for line in include_str!("data/rational.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        if single && fields[1] == "f64" {
            continue;
        }
        let poles: Vec<_> = fields[2]
            .split('|')
            .map(|s| {
                let (p, m) = s.split_once(':').unwrap();
                (p.parse::<f64>().unwrap(), m.parse::<usize>().unwrap())
            })
            .collect();
        let factors: Vec<_> = poles
            .iter()
            .map(|&(p, _)| {
                ApproxPolynomial::monomial_with_options(vec![c(-p), c(1.0)], policy).unwrap()
            })
            .collect();
        let mut denominator =
            ApproxPolynomial::monomial_with_options(vec![c(1.0)], policy).unwrap();
        for (factor, &(_, m)) in factors.iter().zip(&poles) {
            for _ in 0..m {
                denominator = denominator.try_mul(factor).unwrap();
            }
        }
        let inputs: Vec<_> = factors
            .iter()
            .zip(&poles)
            .map(|(f, &(_, m))| (f, m))
            .collect();
        let plan = ApproxPartialFractionPlan::new(&denominator, &inputs).unwrap();
        let values = |s: &str| {
            s.split('|')
                .map(|x| c(x.parse::<f64>().unwrap()))
                .collect::<Vec<_>>()
        };
        let numerator = values(fields[3]);
        let expected = values(fields[4]);
        let result = plan.decompose(&numerator).unwrap();
        let tolerance =
            C::Real::try_from_f64_lossy(fields[if single { 7 } else { 8 }].parse().unwrap())
                .unwrap();
        for (&actual, &oracle) in result.residues().iter().zip(&expected) {
            let scale = oracle.abs();
            let error = (actual - oracle).abs();
            assert!(
                error <= tolerance * scale,
                "{}: {actual:?} vs {oracle:?}",
                fields[0]
            );
        }
        let x = c(fields[5].parse().unwrap());
        let mut value = c(0.0);
        for term in plan.terms() {
            let pole = c(poles[term.factor_index()].0);
            let denominator = (0..term.power()).fold(c(1.0), |v, _| v * (x - pole));
            value = value + result.residues()[term.coefficients().start] / denominator;
        }
        let oracle = c(fields[6].parse().unwrap());
        assert!(
            (value - oracle).abs() <= tolerance * oracle.abs(),
            "evaluation {}",
            fields[0]
        );
        assert!(result.report().unwrap().reconstruction_error() <= tolerance);
    }
}
#[test]
fn rational_oracle_f32() {
    fixture_cases::<f32>(true);
}
#[test]
fn rational_oracle_f64() {
    fixture_cases::<f64>(false);
}
#[test]
fn rational_oracle_complex32() {
    fixture_cases::<Complex<f32>>(true);
}
#[test]
fn rational_oracle_complex64() {
    fixture_cases::<Complex<f64>>(false);
}

#[test]
fn cancellation_and_executor_contracts() {
    use scioxide_core::{CancellationToken, ExecutionContext};
    let policy = PolynomialApproximationPolicy::default();
    let f = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let plan = ApproxPartialFractionPlan::new(&f, &[(&f, 1)]).unwrap();
    let expected = plan.decompose(&[3.0, 2.0]).unwrap();
    let mut output = expected.clone();
    let mut workspace = plan.workspace(2).unwrap();
    #[cfg(feature = "std")]
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(feature = "std")]
    let cx = ExecutionContext::new(&executor);
    #[cfg(not(feature = "std"))]
    let cx = ExecutionContext::sequential();
    for _ in 0..2 {
        plan.decompose_into(&[3.0, 2.0], &mut output, &mut workspace, cx)
            .unwrap();
        assert_eq!(output, expected);
    }
    let token = CancellationToken::new();
    token.cancel();
    let error = plan
        .decompose_into(
            &[3.0, 2.0],
            &mut output,
            &mut workspace,
            cx.with_cancellation(&token),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(output, expected);
}

#[test]
fn factor_order_changes_only_term_order() {
    let policy = PolynomialApproximationPolicy::default();
    let a = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![2.0, 1.0], policy).unwrap();
    let d = a.try_mul(&b).unwrap();
    let forward = ApproxPartialFractionPlan::new(&d, &[(&a, 1), (&b, 1)])
        .unwrap()
        .decompose(&[1.0])
        .unwrap();
    let reversed = ApproxPartialFractionPlan::new(&d, &[(&b, 1), (&a, 1)])
        .unwrap()
        .decompose(&[1.0])
        .unwrap();
    for (&x, &y) in forward
        .residues()
        .iter()
        .zip(reversed.residues().iter().rev())
    {
        assert!((x - y).abs() <= 1e-14);
    }
}

#[test]
fn unrepresentable_half_subnormal_residues_are_rejected() {
    let policy = PolynomialApproximationPolicy::default();
    let a = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![1.0, 1.0], policy).unwrap();
    let d = a.try_mul(&b).unwrap();
    let plan = ApproxPartialFractionPlan::new(&d, &[(&a, 1), (&b, 1)]).unwrap();
    // Both residues are exactly half a minimum subnormal; no scalar output
    // can represent them with an acceptable reconstruction error.
    assert!(matches!(
        (plan.decompose(&[f64::from_bits(1)])).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
}

#[test]
fn long_polynomial_part_reconstructs_with_reserved_storage() {
    let policy = PolynomialApproximationPolicy::default();
    let a = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![1.0, 0.0, 1.0], policy).unwrap();
    let d = a.try_mul(&b).unwrap();
    let polynomial = ApproxPolynomial::monomial_with_options(vec![1.0; 33], policy).unwrap();
    let numerator = polynomial.try_mul(&d).unwrap();
    let plan = ApproxPartialFractionPlan::new(&d, &[(&a, 1), (&b, 1)]).unwrap();
    let result = plan.decompose(numerator.basis().coefficients()).unwrap();
    assert_eq!(result.polynomial(), polynomial.basis().coefficients());
    assert!(result.residues().iter().all(|&x| x == 0.0));
    assert_eq!(result.report().unwrap().reconstruction_error(), 0.0);
}

#[test]
fn one_workspace_serves_every_smaller_plan_in_either_order() {
    use scioxide_core::{Capacity, ExecutionContext};
    let policy = PolynomialApproximationPolicy::default();
    let linear =
        |root: f64| ApproxPolynomial::monomial_with_options(vec![-root, 1.0_f64], policy).unwrap();
    let (a, b, c) = (linear(1.0), linear(2.0), linear(3.0));
    let quadratic = a.try_mul(&b).unwrap();
    let cubic = quadratic.try_mul(&c).unwrap();
    let degree2 = ApproxPartialFractionPlan::new(&quadratic, &[(&a, 1), (&b, 1)]).unwrap();
    let degree3 = ApproxPartialFractionPlan::new(&cubic, &[(&a, 1), (&b, 1), (&c, 1)]).unwrap();
    let plans = [&degree3, &degree2];
    let requirements = plans.map(|plan| plan.workspace_requirements(1).unwrap());
    let cx = ExecutionContext::sequential();
    for (first, second) in [(0, 1), (1, 0)] {
        let mut workspace = PartialFractionWorkspace::new();
        let mut output = ApproxPartialFractions::new();
        for index in [first, second] {
            workspace.try_reserve(&requirements[index]).unwrap();
            output.try_reserve(&requirements[index]).unwrap();
        }
        // Reserving the smaller plan again never shrinks retained capacity.
        workspace.try_reserve(&requirements[1]).unwrap();
        for index in [first, second, first] {
            assert_eq!(
                workspace.check(&requirements[index]).unwrap(),
                Capacity::Sufficient
            );
            plans[index]
                .decompose_into(&[1.0], &mut output, &mut workspace, cx)
                .unwrap();
            // Reused storage must recover the analytic residues. Native owned
            // execution may select different kernels and round differently.
            let expected: &[f64] = if index == 0 {
                &[0.5, -1.0, 0.5]
            } else {
                &[-1.0, 1.0]
            };
            let tolerance = 64.0 * f64::EPSILON;
            assert_eq!(output.polynomial(), &[0.0]);
            assert_eq!(
                output.report().unwrap().rank(),
                plans[index]
                    .decompose(&[1.0])
                    .unwrap()
                    .report()
                    .unwrap()
                    .rank()
            );
            assert_eq!(output.residues().len(), expected.len());
            for (&actual, &expected) in output.residues().iter().zip(expected) {
                assert!((actual - expected).abs() <= tolerance);
            }
            assert!(output.report().unwrap().reconstruction_error() <= tolerance);
        }
    }
}

/// The quotient length `numerator_len - degree` shrinks as the plan degree
/// grows; storage reserved once for the largest plan and longest numerator
/// still serves every smaller plan and numerator, whose quotients are longer.
#[test]
fn storage_for_the_largest_request_serves_longer_quotients_of_smaller_plans() {
    use scioxide_core::{Capacity, ExecutionContext};
    let policy = PolynomialApproximationPolicy::default();
    let linear =
        |root: f64| ApproxPolynomial::monomial_with_options(vec![-root, 1.0_f64], policy).unwrap();
    let (a, b, c) = (linear(1.0), linear(2.0), linear(3.0));
    let quadratic = a.try_mul(&b).unwrap();
    let cubic = quadratic.try_mul(&c).unwrap();
    let degree2 = ApproxPartialFractionPlan::new(&quadratic, &[(&a, 1), (&b, 1)]).unwrap();
    let degree3 = ApproxPartialFractionPlan::new(&cubic, &[(&a, 1), (&b, 1), (&c, 1)]).unwrap();
    let numerator = [1.0, 2.0, 3.0, 4.0, 5.0];
    let largest = degree3.workspace_requirements(numerator.len()).unwrap();
    let mut workspace = degree3.workspace(numerator.len()).unwrap();
    let mut output = ApproxPartialFractions::new();
    output.try_reserve(&largest).unwrap();
    let cx = ExecutionContext::sequential();
    for plan in [&degree2, &degree3, &degree2] {
        for len in [5, 1, 3, 5] {
            assert_eq!(
                workspace
                    .check(&plan.workspace_requirements(len).unwrap())
                    .unwrap(),
                Capacity::Sufficient
            );
            plan.decompose_into(&numerator[..len], &mut output, &mut workspace, cx)
                .unwrap();
            let expected = plan.decompose(&numerator[..len]).unwrap();
            assert_eq!(output.polynomial().len(), expected.polynomial().len());
            for (actual, reference) in [
                (output.polynomial(), expected.polynomial()),
                (output.residues(), expected.residues()),
            ] {
                for (&actual, &reference) in actual.iter().zip(reference) {
                    assert!((actual - reference).abs() <= 64.0 * f64::EPSILON * reference.abs());
                }
            }
            assert!(output.report().unwrap().reconstruction_error() <= 64.0 * f64::EPSILON);
        }
    }
    // (5x⁴ + 4x³ + 3x² + 2x + 1)/((x-1)(x-2)) has the quotient 5x² + 19x + 50.
    degree2
        .decompose_into(&numerator, &mut output, &mut workspace, cx)
        .unwrap();
    for (&actual, expected) in output.polynomial().iter().zip([50.0, 19.0, 5.0]) {
        assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * expected);
    }
}
