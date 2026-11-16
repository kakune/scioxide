//! Sparse evaluation and a tensor-product Bezier surface.
use scioxide_core::{Interval, SciOxideError};
use scioxide_poly::{ApproxSparsePolynomial, Monomial, MonomialOrder, TensorBernstein};
fn main() -> Result<(), SciOxideError> {
    let polynomial = ApproxSparsePolynomial::from_terms(
        2,
        MonomialOrder::GradedLexicographic,
        vec![
            (Monomial::new(&[2, 0])?, 2.0_f64),
            (Monomial::new(&[1, 1])?, 3.),
        ],
    )?;
    println!("p(2,3) = {}", polynomial.evaluate(&[2., 3.])?);
    println!(
        "partial x = {}",
        polynomial.derivative(0)?.evaluate(&[2., 3.])?
    );
    let surface = TensorBernstein::new(
        &[1, 1],
        vec![Interval::new(0., 1.)?; 2],
        3,
        vec![0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1.],
    )?;
    println!("surface point = {:?}", surface.evaluate(&[0.25, 0.5])?);
    let (left, right) = surface.subdivide(0, 0.5)?;
    println!(
        "ordinary child control boxes: {:?}, {:?}",
        left.control_bounds()?.coordinates(),
        right.control_bounds()?.coordinates()
    );
    Ok(())
}
