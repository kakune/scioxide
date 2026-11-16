//! ProductRule correctness, work-accounting, dispatch, and allocation regressions.

#![allow(clippy::unwrap_used)]

use scioxide_integrate::GaussianRulePlan;
use std::sync::atomic::AtomicUsize as Cell;

use scioxide_core::{Interval, SciOxideError};
use scioxide_integrate::{FixedRule, FixedRuleNode, IntegrationTermination, ProductRule};

fn bounds(intervals: &[(f64, f64)]) -> Vec<Interval<f64>> {
    intervals
        .iter()
        .map(|&(lower, upper)| Interval::new(lower, upper).unwrap())
        .collect()
}

fn unit_bounds(dimension: usize) -> Vec<Interval<f64>> {
    (0..dimension)
        .map(|_| Interval::new(0.0, 1.0).unwrap())
        .collect()
}

#[test]
fn dimensions_one_through_four_have_exact_tensor_callback_counts() {
    let rule = GaussianRulePlan::gauss_legendre(5).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());

    for dimension in 1..=4 {
        let callbacks = Cell::new(0_usize);
        let integrand = |point: &[f64]| {
            callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(point
                .iter()
                .map(|coordinate| coordinate * coordinate)
                .sum::<f64>())
        };
        let result = product_rule
            .integrate(&integrand, &unit_bounds(dimension))
            .unwrap();
        let expected_callbacks = 5_usize.pow(dimension as u32);

        assert!((result.0 - dimension as f64 / 3.0).abs() < 2.0e-13);
        assert_eq!(result.1.evaluations(), expected_callbacks);
        assert_eq!(
            callbacks.load(core::sync::atomic::Ordering::Relaxed),
            expected_callbacks
        );
    }
}

#[test]
fn asymmetric_bounds_preserve_tensor_product_mathematics() {
    let rule = GaussianRulePlan::gauss_legendre(4).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = bounds(&[(-2.0, 1.0), (0.5, 2.0), (3.0, 5.0)]);
    let integrand = |point: &[f64]| Ok(point[0] * point[1] * point[1] * (point[2] + 1.0));

    let result = product_rule.integrate(&integrand, &domain).unwrap();

    assert!((result.0 - (-39.375)).abs() < 2.0e-12);
    assert_eq!(result.1.evaluations(), 64);
}

#[test]
fn recursive_coordinate_borrows_do_not_overlap() {
    let rule = GaussianRulePlan::gauss_legendre(3).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = bounds(&[(-2.0, -1.0), (1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let callbacks = Cell::new(0_usize);
    let integrand = |point: &[f64]| {
        assert!(point.iter().zip(&domain).all(|(&coordinate, interval)| {
            coordinate >= interval.lower() && coordinate <= interval.upper()
        }));
        callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };

    let result = product_rule.integrate(&integrand, &domain).unwrap();

    assert!((result.0 - 1.0).abs() < 2.0e-13);
    assert_eq!(callbacks.load(core::sync::atomic::Ordering::Relaxed), 81);
    assert_eq!(result.1.evaluations(), 81);
}

#[test]
fn degenerate_and_reversed_bounds_follow_interval_policy() {
    let rule = GaussianRulePlan::gauss_legendre(5).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = bounds(&[(0.0, 1.0), (2.0, 2.0), (-1.0, 3.0)]);
    let callbacks = Cell::new(0_usize);
    let integrand = |_point: &[f64]| {
        callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };

    let result = product_rule.integrate(&integrand, &domain).unwrap();

    assert_eq!(result.0, 0.0);
    assert_eq!(result.1.evaluations(), 0);
    assert_eq!(callbacks.load(core::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(
        result.1.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert!(matches!(
        (Interval::<f64>::new(1.0, -1.0))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn callback_errors_and_non_finite_values_stop_at_the_first_leaf() {
    let rule = GaussianRulePlan::gauss_legendre(3).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = unit_bounds(2);

    let error_callbacks = Cell::new(0_usize);
    let failing = |_point: &[f64]| -> Result<f64, SciOxideError> {
        error_callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("point"),
            scioxide_core::DomainReason::Undefined,
            "sentinel callback error",
        ))
    };
    assert!(matches!(
        (product_rule.integrate(&failing, &domain))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(
        error_callbacks.load(core::sync::atomic::Ordering::Relaxed),
        1
    );

    let non_finite_callbacks = Cell::new(0_usize);
    let non_finite = |_point: &[f64]| {
        non_finite_callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(f64::NAN)
    };
    assert!(matches!(
        (product_rule.integrate(&non_finite, &domain))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(
        non_finite_callbacks.load(core::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn concrete_and_erased_integrands_have_identical_values_and_counts() {
    let rule = GaussianRulePlan::gauss_legendre(4).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = bounds(&[(-1.0, 2.0), (2.0, 4.0)]);
    let integrand = |point: &[f64]| Ok(point[0] + point[1]);
    let dynamic_integrand: &(dyn Fn(&[f64]) -> Result<f64, SciOxideError> + Sync) = &integrand;

    let static_result = product_rule.integrate(&integrand, &domain).unwrap();
    let dynamic_result = product_rule.integrate(dynamic_integrand, &domain).unwrap();

    assert_eq!(static_result.0, dynamic_result.0);
    assert_eq!(static_result.1.evaluations(), 16);
    assert_eq!(
        static_result.1.evaluations(),
        dynamic_result.1.evaluations()
    );
}

#[test]
fn tensor_work_overflow_is_reported_before_the_user_callback() {
    let rule = GaussianRulePlan::gauss_legendre(2).unwrap();
    let product_rule = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let domain = unit_bounds(usize::BITS as usize);
    let callbacks = Cell::new(0_usize);
    let integrand = |_point: &[f64]| {
        callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };

    let result = product_rule.integrate(&integrand, &domain);

    assert!(
        matches!(&result, Err(error) if match error.view() { scioxide_core::SciOxideErrorView::SizeOverflow {  context: _view_context } => _view_context == scioxide_core::quantity!("tensor_cardinality"), _ => false }),
        "{result:?}"
    );
    assert_eq!(callbacks.load(core::sync::atomic::Ordering::Relaxed), 0);
}

struct OneNodeRule {
    metadata_calls: Cell,
}

impl FixedRule<f64> for OneNodeRule {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "OneNodeRule", 1)
    }

    fn node_count(&self) -> usize {
        1
    }

    fn map_nodes_into(
        &self,
        interval: Interval<f64>,
        destination: &mut [FixedRuleNode<f64>],
    ) -> Result<(), SciOxideError> {
        if destination.len() != 1 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("index"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "one-node rule only exposes index zero",
            ));
        }
        self.metadata_calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        destination[0] = FixedRuleNode::new(
            interval.lower() + (interval.upper() - interval.lower()) * 0.5,
            interval.upper() - interval.lower(),
            None,
        );
        Ok(())
    }
}

#[test]
fn object_safe_metadata_traverses_high_dimension_without_recursion() {
    let dimension = 4096;
    let rule = OneNodeRule {
        metadata_calls: Cell::new(0),
    };
    let erased: &dyn FixedRule<f64> = &rule;
    let product = ProductRule::new(erased, scioxide_integrate::ProductOptions::default());
    let domain = unit_bounds(dimension);
    let callbacks = Cell::new(0_usize);
    let integrand = |point: &[f64]| {
        callbacks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        assert_eq!(point.len(), dimension);
        Ok(1.0)
    };
    let mut workspace = product.workspace((domain.len(), 1)).unwrap();
    let result = (|| {
        let mut value = domain
            .first()
            .map(|b| b.lower())
            .unwrap_or_else(|| <_ as scioxide_core::AlgebraicScalar>::ZERO);
        let report = product.integrate_into(
            &integrand,
            &domain,
            &mut value,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok::<_, scioxide_core::SciOxideError>((value, report))
    })()
    .unwrap();

    assert_eq!(
        rule.metadata_calls
            .load(core::sync::atomic::Ordering::Relaxed),
        dimension
    );
    assert_eq!(callbacks.load(core::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(result.1.evaluations(), 1);
    assert_eq!(result.0, 1.0);
    assert_eq!(
        result.1.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
}
