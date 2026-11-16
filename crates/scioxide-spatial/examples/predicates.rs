//! Exact topology decisions without constructing approximate circles/spheres.
use scioxide_core::SciOxideError;
use scioxide_spatial::{Point2, PredicateSign, incircle, insphere, orient2d};

fn main() -> Result<(), SciOxideError> {
    let a = Point2::new([1.0_f64, 0.0])?;
    let b = Point2::new([0.0, 1.0])?;
    let c = Point2::new([-1.0, 0.0])?;
    assert_eq!(
        orient2d(a.as_array(), b.as_array(), c.as_array())?,
        PredicateSign::Positive
    );
    assert_eq!(
        incircle(a.as_array(), b.as_array(), c.as_array(), &[0.0, -1.0])?,
        PredicateSign::Zero
    );
    assert_eq!(
        incircle(a.as_array(), b.as_array(), c.as_array(), &[0.0, 0.0])?,
        PredicateSign::Positive
    );
    assert_eq!(
        insphere(
            &[1.0_f32, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[-1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0]
        )?,
        PredicateSign::Positive
    );
    println!("Circle boundary is exactly degenerate; circle and sphere centers are inside.");
    Ok(())
}
