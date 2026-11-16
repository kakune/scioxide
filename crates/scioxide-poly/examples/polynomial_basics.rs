//! Exact-domain and diagnostic approximate polynomial basics.

use scioxide_core::{RankThreshold, SciOxideError, Tolerance};
use scioxide_linalg::RankPolicy;
use scioxide_poly::{
    ApproxPolynomial, ExactPolynomial, PolynomialApproximationPolicy, approximate_gcd,
};

// Illustrative bounded rational field supplied by the caller, shared with tests.
#[path = "../tests/support/exact_field.rs"]
mod exact_field;
use exact_field::Q;

fn main() -> Result<(), SciOxideError> {
    let exact = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    let factor = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    let (quotient, remainder) = exact.div_rem(&factor)?;
    assert_eq!(quotient.coefficients(), &[Q(1, 1), Q(1, 1)]);
    assert!(remainder.is_zero());
    println!("exact quotient at 1/2 = {:?}", quotient.evaluate(Q(1, 2)));

    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(0.0_f64, 1.0e-10)?,
        RankPolicy::Threshold(RankThreshold::new(0.0, 1.0e-10)?),
    );
    let polynomial = ApproxPolynomial::from_roots_with_options(&[-2.0, 1.0, 3.0], policy)?;
    let evaluation = polynomial.evaluate_with_derivative(0.5)?;
    println!(
        "p(0.5) = {}, p'(0.5) = {}, backward error <= {}",
        evaluation.value(),
        evaluation.derivative(),
        evaluation.condition().backward_error()
    );

    let shared = ApproxPolynomial::from_roots_with_options(&[1.0], policy)?;
    let left = shared.try_mul(&ApproxPolynomial::from_roots_with_options(&[-2.0], policy)?)?;
    let right = shared.try_mul(&ApproxPolynomial::from_roots_with_options(&[3.0], policy)?)?;
    let gcd = approximate_gcd(&left, &right)?;
    println!(
        "effective GCD degree {}, Sylvester rank {:?}, backward error {}",
        gcd.report().effective_degree(),
        gcd.report().sylvester_rank().map(|rank| rank.rank()),
        gcd.report().backward_error()
    );
    Ok(())
}
