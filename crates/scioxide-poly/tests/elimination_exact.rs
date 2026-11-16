//! Independent determinant and exact-identity coverage.
#![allow(clippy::unwrap_used)]
#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;
use scioxide_poly::{ExactField, ExactPolynomial, exact_discriminant, exact_subresultants};
/// The resultant is read from the subresultant sequence; there is no separate entry point.
fn exact_resultant<C: ExactField>(
    left: &ExactPolynomial<C>,
    right: &ExactPolynomial<C>,
) -> Result<C, scioxide_core::SciOxideError> {
    exact_subresultants(left, right).map(|sequence| sequence.resultant().clone())
}
fn p(c: &[i128]) -> ExactPolynomial<Q> {
    ExactPolynomial::new(c.iter().map(|&x| Q(x, 1)).collect()).unwrap()
}
// Leibniz/cofactor oracle: no polynomial division or production PRS recurrence.
fn det(a: &[Vec<Q>]) -> Q {
    if a.is_empty() {
        return Q(1, 1);
    }
    a[0].iter().enumerate().fold(Q(0, 1), |s, (j, &c)| {
        let minor: Vec<_> = a[1..]
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .filter_map(|(k, &v)| (k != j).then_some(v))
                    .collect()
            })
            .collect();
        s + Q(if j % 2 == 0 { 1 } else { -1 }, 1) * c * det(&minor)
    })
}
fn oracle(a: &ExactPolynomial<Q>, b: &ExactPolynomial<Q>) -> Q {
    if a.is_zero() || b.is_zero() {
        return Q(0, 1);
    }
    let (m, n) = (a.degree(), b.degree());
    let mut matrix = vec![vec![Q(0, 1); m + n]; m + n];
    for (r, row) in matrix.iter_mut().enumerate() {
        let (poly, shift) = if r < n { (a, r) } else { (b, r - n) };
        for (j, &c) in poly.coefficients().iter().rev().enumerate() {
            row[shift + j] = c;
        }
    }
    det(&matrix)
}
#[test]
fn independent_determinants_and_swap_signs() {
    let fixtures = [
        p(&[0]),
        p(&[3]),
        p(&[-2, 1]),
        p(&[1, 2]),
        p(&[1, 0, 1]),
        p(&[-1, 0, 1]),
        p(&[2, -1, 0, 2]),
    ];
    for a in &fixtures {
        for b in &fixtures {
            assert_eq!(exact_resultant(a, b).unwrap(), oracle(a, b), "{a:?}, {b:?}");
        }
    }
}
#[test]
fn defective_prs_and_discriminants() {
    let sequence = exact_subresultants(&p(&[1, 0, 1]), &p(&[-1, 0, 1])).unwrap();
    assert_eq!(
        sequence
            .polynomials()
            .iter()
            .map(|p| p.degree())
            .collect::<Vec<_>>(),
        [2, 2, 0]
    );
    assert_eq!(sequence.polynomials()[2], p(&[-2]));
    assert_eq!(sequence.scalar_subresultants(), [Q(1, 1), Q(1, 1), Q(4, 1)]);
    assert_eq!(exact_discriminant(&p(&[1, 2, 3])).unwrap(), Q(-8, 1));
    assert_eq!(exact_discriminant(&p(&[1, -2, 1])).unwrap(), Q(0, 1));
    assert_eq!(exact_discriminant(&p(&[-1, 0, 0, 1])).unwrap(), Q(-27, 1));
    assert_eq!(exact_discriminant(&p(&[2])).unwrap(), Q(1, 1));
    assert!(exact_discriminant(&p(&[0])).is_err());
}
#[test]
fn multiplicativity_and_gcd_agree() {
    let (a, b, c) = (p(&[-2, 1]), p(&[1, 0, 1]), p(&[3, 2]));
    assert_eq!(
        exact_resultant(&a.try_mul(&b).unwrap(), &c).unwrap(),
        exact_resultant(&a, &c).unwrap() * exact_resultant(&b, &c).unwrap()
    );
    let left = a.try_mul(&b).unwrap();
    let right = a.try_mul(&c).unwrap();
    let seq = exact_subresultants(&left, &right).unwrap();
    assert_eq!(seq.resultant(), &Q(0, 1));
    assert_eq!(
        seq.polynomials().last().unwrap().monic().unwrap(),
        left.gcd(&right).unwrap()
    );
}

#[path = "support/prime_field.rs"]
mod prime_field;
use prime_field::Fp;
#[test]
fn positive_characteristic_derivative_degree_loss() {
    let p = |v: &[bool]| {
        ExactPolynomial::new(v.iter().map(|&x| Fp::<2>(u64::from(x))).collect()).unwrap()
    };
    // In characteristic two, D(ax²+bx+c)=b², including a vanished x term in p'.
    assert_eq!(
        exact_discriminant(&p(&[true, true, true])).unwrap(),
        Fp::<2>(1)
    );
    assert_eq!(
        exact_discriminant(&p(&[true, false, true])).unwrap(),
        Fp::<2>(0)
    );
    let a = p(&[true, true, true]);
    let b = p(&[true, true]);
    assert_eq!(exact_resultant(&a, &b).unwrap(), Fp::<2>(1));
}

#[test]
fn bounded_generated_determinant_fixtures() {
    for seed in 1..=12i128 {
        let a = p(&[seed % 3 - 1, 2 - seed % 5, seed % 2 + 1, 1]);
        let b = p(&[seed % 5 - 2, seed % 3 - 1, 2]);
        assert_eq!(exact_resultant(&a, &b).unwrap(), oracle(&a, &b));
        let scaled = p(&[3 * (seed % 5 - 2), 3 * (seed % 3 - 1), 6]);
        assert_eq!(
            exact_resultant(&a, &scaled).unwrap(),
            Q(27, 1) * oracle(&a, &b)
        );
    }
}

#[test]
fn nonunit_leading_coefficient_with_multiple_missing_derivative_degrees() {
    // In GF(3), p'=1, Res(p,1)=1, and disc(p)=-2=1.
    let cubic = ExactPolynomial::new(vec![Fp::<3>(1), Fp(1), Fp(0), Fp(2)]).unwrap();
    assert_eq!(cubic.derivative().unwrap().coefficients(), &[Fp(1)]);
    assert_eq!(exact_discriminant(&cubic).unwrap(), Fp(1));
    // In GF(5), p'=1 and the missing-degree correction is lc(p)^3=8=3.
    let quintic =
        ExactPolynomial::new(vec![Fp::<5>(1), Fp(1), Fp(0), Fp(0), Fp(0), Fp(2)]).unwrap();
    assert_eq!(quintic.derivative().unwrap().coefficients(), &[Fp(1)]);
    assert_eq!(exact_discriminant(&quintic).unwrap(), Fp(3));
}
