//! Certified endpoint containment and a preserved pole split.
use scioxide_core::SciOxideError;
use scioxide_validated::{DecoratedInterval, EnclosureStatus};
fn main() -> Result<(), SciOxideError> {
    let input = DecoratedInterval::new(1.0_f64, 2.0)?;
    let exponential = input.exp()?;
    assert_eq!(exponential.status(), EnclosureStatus::Defined);
    assert!(exponential.contains(3.0));
    let tangent = input.tan()?;
    assert_eq!(tangent.status(), EnclosureStatus::PartiallyDefined);
    assert_eq!(tangent.pieces().len(), 2);
    assert!(!tangent.contains(0.0));
    println!("exp([1,2]): {:?}", exponential.pieces());
    println!("tan([1,2]) branches: {:?}", tangent.pieces());
    Ok(())
}
