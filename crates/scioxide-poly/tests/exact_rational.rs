//! Exact field references for rational algebra and factor-adic residues.
#![allow(clippy::unwrap_used)]
use scioxide_core::SciOxideErrorKind;
use scioxide_poly::{ExactPartialFractionPlan, ExactPolynomial, ExactRationalFunction};

#[path = "support/exact_field.rs"]
mod exact_field;
#[path = "support/truncated_field.rs"]
mod truncated_field;
use exact_field::Q;

fn polynomial(coefficients: &[i128]) -> ExactPolynomial<Q> {
    ExactPolynomial::new(coefficients.iter().map(|&value| Q(value, 1)).collect()).unwrap()
}

#[test]
fn closed_form_residues_for_simple_and_repeated_factors() {
    let linear = polynomial(&[-1, 1]);
    let other = polynomial(&[2, 1]);
    let quadratic = polynomial(&[1, 0, 1]);
    // 3x/((x-1)(x+2)) = 1/(x-1) + 2/(x+2).
    let denominator = polynomial(&[-2, 1, 1]);
    let result = ExactPartialFractionPlan::new(&denominator, &[(&linear, 1), (&other, 1)])
        .unwrap()
        .decompose(&polynomial(&[0, 3]))
        .unwrap();
    assert_eq!(result.polynomial(), &polynomial(&[0]));
    for (term, (factor, residue)) in result.terms().iter().zip([(&linear, 1), (&other, 2)]) {
        assert_eq!(term.factor(), factor);
        assert_eq!(term.power(), 1);
        assert_eq!(term.numerator(), &polynomial(&[residue]));
    }

    // (4-x-4x²+5x³)/((x-1)²(x²+1))
    // = 1/(x-1) + 2/(x-1)² + (3+4x)/(x²+1).
    let denominator = polynomial(&[1, -2, 2, -2, 1]);
    let result = ExactPartialFractionPlan::new(&denominator, &[(&linear, 2), (&quadratic, 1)])
        .unwrap()
        .decompose(&polynomial(&[4, -1, -4, 5]))
        .unwrap();
    for (term, (power, numerator)) in result.terms().iter().zip([
        (1, polynomial(&[1])),
        (2, polynomial(&[2])),
        (1, polynomial(&[3, 4])),
    ]) {
        assert_eq!(term.power(), power);
        assert_eq!(term.numerator(), &numerator);
    }

    // (8+10x+3x²+4x³)/(1+x²)² = (3+4x)/(1+x²) + (5+6x)/(1+x²)².
    let denominator = polynomial(&[1, 0, 2, 0, 1]);
    let plan = ExactPartialFractionPlan::new(&denominator, &[(&quadratic, 2)]).unwrap();
    for (numerator, part) in [
        (polynomial(&[8, 10, 3, 4]), polynomial(&[0])),
        // Add (x+2)(1+x²)² using hand-expanded coefficients.
        (polynomial(&[10, 11, 7, 6, 2, 1]), polynomial(&[2, 1])),
    ] {
        let result = plan.decompose(&numerator).unwrap();
        assert_eq!(result.polynomial(), &part);
        assert_eq!(result.terms().len(), 2);
        for (term, (power, coefficients)) in result.terms().iter().zip([(1, [3, 4]), (2, [5, 6])]) {
            assert_eq!(term.factor(), &quadratic);
            assert_eq!(term.power(), power);
            assert_eq!(term.numerator(), &polynomial(&coefficients));
        }
    }
}

#[test]
fn exact_rational_arithmetic_has_independent_values() {
    let a = ExactRationalFunction::new(polynomial(&[2, 2]), polynomial(&[4, 2])).unwrap();
    let b = ExactRationalFunction::new(polynomial(&[3]), polynomial(&[1, 1])).unwrap();
    for (actual, expected) in [
        (a.try_add(&b).unwrap(), Q::new(31, 20)),
        (a.try_sub(&b).unwrap(), Q::new(1, 20)),
        (a.try_mul(&b).unwrap(), Q::new(3, 5)),
        (a.try_div(&b).unwrap(), Q::new(16, 15)),
        (a.reciprocal().unwrap(), Q::new(5, 4)),
    ] {
        assert_eq!(actual.evaluate(Q(3, 1)).unwrap(), expected);
    }
    let zero = polynomial(&[0]);
    let nonzero = polynomial(&[2, 4, 2]);
    assert_eq!(zero.gcd(&zero).unwrap(), zero);
    assert_eq!(zero.gcd(&nonzero).unwrap(), polynomial(&[1, 2, 1]));
    assert_eq!(zero.lcm(&nonzero).unwrap(), zero);
    assert_eq!(nonzero.lcm(&zero).unwrap(), zero);
    assert_eq!(zero.lcm(&zero).unwrap(), zero);
    let original = nonzero.clone();
    assert_eq!(
        nonzero.div_rem(&zero).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(nonzero, original);
    assert_eq!(
        ExactRationalFunction::new(nonzero.clone(), zero.clone())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(nonzero, original);
    assert_eq!(zero, polynomial(&[0]));
}

#[test]
fn inconsistent_field_division_returns_breakdown_with_bounded_work() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    use std::sync::atomic::Ordering;
    use truncated_field::{DIVISIONS, Z};
    let a = ExactPolynomial::new(vec![Z(1), Z(0), Z(1)]).unwrap();
    let b = ExactPolynomial::new(vec![Z(0), Z(2)]).unwrap();
    for error in [
        a.div_rem(&b).unwrap_err(),
        a.gcd(&b).unwrap_err(),
        a.lcm(&b).unwrap_err(),
        scioxide_poly::exact_subresultants(&a, &b).unwrap_err(),
        ExactRationalFunction::new(a, b).unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Breakdown,
                ..
            }
        ));
    }
    assert_eq!(DIVISIONS.load(Ordering::Relaxed), 5);
}

#[test]
fn reciprocal_preserves_canonical_coprime_normalization() {
    let rational = ExactRationalFunction::new(polynomial(&[2, 4]), polynomial(&[3, 3])).unwrap();
    let reciprocal = rational.reciprocal().unwrap();
    assert_eq!(
        reciprocal.numerator().coefficients(),
        [Q::new(3, 4), Q::new(3, 4)]
    );
    assert_eq!(
        reciprocal.denominator().coefficients(),
        [Q::new(1, 2), Q(1, 1)]
    );
    assert_eq!(reciprocal.reciprocal().unwrap(), rational);
    for value in [-3, 1, 4] {
        let constant = ExactRationalFunction::new(polynomial(&[value]), polynomial(&[1])).unwrap();
        let reciprocal = constant.reciprocal().unwrap();
        assert_eq!(reciprocal.numerator().coefficients(), [Q::new(1, value)]);
        assert_eq!(reciprocal.denominator(), &polynomial(&[1]));
    }
    let zero = ExactRationalFunction::new(polynomial(&[0]), polynomial(&[2, 1])).unwrap();
    assert_eq!(
        zero.reciprocal().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn repeated_factor_powers_reconstruct_odd_and_even_multiplicities() {
    let factor = polynomial(&[0, 1]);
    for multiplicity in [7, 8] {
        let mut coefficients = vec![Q(0, 1); multiplicity + 1];
        coefficients[multiplicity] = Q(1, 1);
        let denominator = ExactPolynomial::new(coefficients).unwrap();
        let numerator = ExactPolynomial::new(vec![Q(1, 1); multiplicity]).unwrap();
        let plan = ExactPartialFractionPlan::new(&denominator, &[(&factor, multiplicity)]).unwrap();
        let result = plan.decompose(&numerator).unwrap();
        assert_eq!(result.terms().len(), multiplicity);
        for (index, term) in result.terms().iter().enumerate() {
            assert_eq!(term.power(), index + 1);
            assert_eq!(term.numerator(), &polynomial(&[1]));
        }
        assert_eq!(
            result.reconstruct().unwrap(),
            ExactRationalFunction::new(numerator, denominator).unwrap()
        );
    }
}
